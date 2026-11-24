// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// Precision selects the accumulation precision used by a dispatched matrix
// operation.
type Precision uint8

const (
	// PrecisionStandard uses the element type's native precision. Float32
	// products and sums are evaluated in float32 and float64 products and sums
	// in float64.
	PrecisionStandard Precision = iota

	// PrecisionHigh uses an exact super-accumulator for the reduction and
	// rounds only the final dot product to the element type. It is intended
	// for strongly cancelling inputs.
	PrecisionHigh
)

// String returns the canonical name of the precision tier.
func (p Precision) String() string {
	switch p {
	case PrecisionStandard:
		return "standard"
	case PrecisionHigh:
		return "high"
	default:
		return fmt.Sprintf("unknown-precision(%d)", uint8(p))
	}
}

// ParsePrecision parses "standard" or "high".
func ParsePrecision(s string) (Precision, error) {
	switch s {
	case "standard", "native":
		return PrecisionStandard, nil
	case "high", "accurate":
		return PrecisionHigh, nil
	default:
		return 0, fmt.Errorf("mat: unknown dispatch precision %q", s)
	}
}

// Route identifies the implementation route selected by the dispatcher.
type Route uint8

const (
	RouteUnknown Route = iota
	RouteGEMMDirect
	RouteGEMMWorkspace
	RouteSymmetric
	RouteTriangularMul
	RouteRankK
	RouteTriSolve
)

// String returns a stable kebab-case route name.
func (r Route) String() string {
	switch r {
	case RouteGEMMDirect:
		return "gemm-direct"
	case RouteGEMMWorkspace:
		return "gemm-workspace"
	case RouteSymmetric:
		return "symmetric"
	case RouteTriangularMul:
		return "triangular-mul"
	case RouteRankK:
		return "rank-k"
	case RouteTriSolve:
		return "triangular-solve"
	default:
		return "unknown"
	}
}

// Chunk records the planned M, N and K tile sizes. A zero component means that
// the corresponding matrix dimension is zero and there is no tile to execute.
type Chunk struct {
	M int
	N int
	K int
}

// RoutePrecisionKey is the composite counter key.
type RoutePrecisionKey struct {
	Route     Route
	Precision Precision
}

// RoutePrecisionCounts contains counters for one route/precision pair.
type RoutePrecisionCounts struct {
	Calls    int64
	Degraded int64
	Failures int64
}

// DispatchCounts is a consistent snapshot of dispatcher counters.
type DispatchCounts struct {
	Calls            int64
	Degraded         int64
	Failures         int64
	ByRoute          map[Route]int64
	ByPrecision      map[Precision]int64
	ByRoutePrecision map[RoutePrecisionKey]RoutePrecisionCounts
}

// HistoryPage is the result of querying the bounded decision ring.
type HistoryPage struct {
	// Records are in chronological order, oldest retained first.
	Records []Decision

	// Total is the number of accepted dispatch decisions since the dispatcher
	// was created or its observability state was reset. It can exceed Capacity.
	Total uint64

	// Capacity is the maximum number of retained records.
	Capacity int
}

// Decision is the observable, immutable record of one dispatch.
type Decision struct {
	ID uint64

	// Op is the operation family, for example "gemm", "symmetric-mul",
	// "triangular-mul", "rank-k" or "triangular-solve".
	Op string

	Route     Route
	Precision Precision

	M int
	N int
	K int

	TransA bool
	TransB bool

	// Chunk is the planned maximum tile in each direction.
	Chunk Chunk

	// Workers is the maximum number of concurrent execution lanes. It includes
	// the calling goroutine when Goroutines is Workers-1.
	Workers int

	// Goroutines is the number of helper goroutines started. Small jobs have
	// Goroutines == 0 and execute entirely on the caller's goroutine.
	Goroutines int

	// KPartitions is the number of deterministic ordered reduction partitions.
	KPartitions int

	// WorkUnits is an implementation-defined amount of work, currently the
	// estimated number of scalar floating-point operations.
	WorkUnits int64

	// WorkspaceBytes is the extra memory reserved for this call.
	WorkspaceBytes int64

	// MemoryBudget is the per-call budget visible when the call was planned.
	MemoryBudget int64

	Degraded       bool
	FallbackReason string

	// Accumulator describes the numerical reduction implementation.
	Accumulator string

	// Specialized is true for structure-aware routes and false when a generic
	// multiplication route is used.
	Specialized bool

	OK    bool
	Error string

	At time.Time
}

const (
	// DefaultHistorySize is the default decision-ring capacity.
	DefaultHistorySize = 64

	defaultChunkM = 64
	defaultChunkN = 64
	defaultChunkK = 256

	// deterministicKPartition keeps reduction association stable even when K
	// work is distributed across goroutines. It is intentionally independent of
	// the configurable cache tile Chunk.K.
	deterministicKPartition = 1024

	// parallelWorkThreshold avoids goroutine startup for tiny products.
	parallelWorkThreshold = 4096
)

// Fallback reasons.
const (
	FallbackMemoryBudget        = "memory-budget"
	FallbackWorkspaceAllocation = "workspace-allocation"
	FallbackHighUnsupported     = "high-precision-unsupported"
)

// ErrHighPrecisionUnsupported is returned by a structured entry that has no
// high-precision implementation. Callers must not interpret a nil error as
// having run the requested tier in that case.
var ErrHighPrecisionUnsupported = errors.New("mat: high precision is not supported for this dispatch route")

// DispatchConfig controls planning, resource limits and observability.
//
// The zero value is not a valid configuration. Obtain a usable configuration
// with DefaultDispatchConfig and then modify it.
type DispatchConfig struct {
	// ChunkM, ChunkN and ChunkK are positive tile sizes. Values larger than a
	// dimension are clamped to that dimension. Non-positive values are invalid
	// so that an accidentally unset field is reported immediately.
	ChunkM int
	ChunkN int
	ChunkK int

	// AdaptiveChunks allows the planner to shrink the configured tile sizes to
	// fit MemoryBudget. The configured values remain the upper bounds. Set it
	// false when all three chunk sizes must be treated as fixed requests.
	AdaptiveChunks bool

	// MaxWorkers limits execution lanes for one call. It must be positive.
	MaxWorkers int

	// MemoryBudget limits extra workspace bytes for one dispatched operation.
	// Zero permits only routes that need no planned extra workspace. Negative
	// values are invalid.
	MemoryBudget int64

	// HistorySize is the ring capacity. Zero disables retained records while
	// still keeping counters. Negative values are invalid.
	HistorySize int

	// Collect enables decision records. Counters are maintained even when
	// collection is disabled. Collection never changes numerical results.
	Collect bool

	// DefaultPrecision is used when a call does not provide an override.
	DefaultPrecision Precision
}

// DefaultDispatchConfig returns the validated default configuration.
func DefaultDispatchConfig() DispatchConfig {
	return DispatchConfig{
		ChunkM:           defaultChunkM,
		ChunkN:           defaultChunkN,
		ChunkK:           defaultChunkK,
		AdaptiveChunks:   true,
		MaxWorkers:       runtime.GOMAXPROCS(0),
		MemoryBudget:     0,
		HistorySize:      DefaultHistorySize,
		Collect:          true,
		DefaultPrecision: PrecisionStandard,
	}
}

func (c DispatchConfig) validated() (DispatchConfig, error) {
	out := c
	if c.ChunkM <= 0 {
		return out, fmt.Errorf("mat: dispatch configuration ChunkM=%d is invalid: must be positive", c.ChunkM)
	}
	if c.ChunkN <= 0 {
		return out, fmt.Errorf("mat: dispatch configuration ChunkN=%d is invalid: must be positive", c.ChunkN)
	}
	if c.ChunkK <= 0 {
		return out, fmt.Errorf("mat: dispatch configuration ChunkK=%d is invalid: must be positive", c.ChunkK)
	}
	if c.MaxWorkers <= 0 {
		return out, fmt.Errorf("mat: dispatch configuration MaxWorkers=%d is invalid: must be positive", c.MaxWorkers)
	}
	if c.MemoryBudget < 0 {
		return out, fmt.Errorf("mat: dispatch configuration MemoryBudget=%d is too low: must be >= 0", c.MemoryBudget)
	}
	if c.HistorySize < 0 {
		return out, fmt.Errorf("mat: dispatch configuration HistorySize=%d is invalid: must be >= 0", c.HistorySize)
	}
	if c.DefaultPrecision != PrecisionStandard && c.DefaultPrecision != PrecisionHigh {
		return out, fmt.Errorf("mat: dispatch configuration DefaultPrecision=%s is unknown", c.DefaultPrecision)
	}
	return out, nil
}

// Dispatcher owns the single source of truth for configuration, decisions,
// counters and reusable workspaces. A Dispatcher is safe for concurrent use.
type Dispatcher struct {
	mu      sync.Mutex
	cfg     DispatchConfig
	records []Decision
	next    int
	filled  bool
	total   uint64
	counts  DispatchCounts
	pool64  workspacePool[float64]
	pool32  workspacePool[float32]
	exact   exactWorkspacePool

	// workspaceAllocErr is an internal failure-injection hook for the
	// allocation-failure degradation path. It is not part of the public API.
	workspaceAllocErr error
}

// NewDispatcher validates cfg and creates an independent dispatcher.
func NewDispatcher(cfg DispatchConfig) (*Dispatcher, error) {
	valid, err := cfg.validated()
	if err != nil {
		return nil, err
	}
	d := &Dispatcher{cfg: valid}
	d.records = make([]Decision, valid.HistorySize)
	d.counts.ByRoute = make(map[Route]int64)
	d.counts.ByPrecision = make(map[Precision]int64)
	d.counts.ByRoutePrecision = make(map[RoutePrecisionKey]RoutePrecisionCounts)
	return d, nil
}

// MustNewDispatcher is like NewDispatcher and panics on invalid configuration.
func MustNewDispatcher(cfg DispatchConfig) *Dispatcher {
	d, err := NewDispatcher(cfg)
	if err != nil {
		panic(err)
	}
	return d
}

var defaultDispatcher = MustNewDispatcher(DefaultDispatchConfig())

// DefaultDispatcher returns the process-wide default dispatcher.
func DefaultDispatch() *Dispatcher { return defaultDispatcher }

// ConfigureDispatch validates and atomically replaces the default dispatcher's
// configuration. Invalid configuration leaves the previous configuration in
// effect.
func ConfigureDispatch(cfg DispatchConfig) error {
	return defaultDispatcher.Configure(cfg)
}

// Configure validates and replaces this dispatcher's configuration. Calls that
// have already started continue with their own snapshot.
func (d *Dispatcher) Configure(cfg DispatchConfig) error {
	valid, err := cfg.validated()
	if err != nil {
		return err
	}
	d.mu.Lock()
	old := d.cfg
	oldPage := d.historyLocked(0)
	d.cfg = valid
	if valid.HistorySize != old.HistorySize {
		newRecords := make([]Decision, valid.HistorySize)
		kept := 0
		if valid.HistorySize > 0 {
			start := 0
			if len(oldPage) > valid.HistorySize {
				start = len(oldPage) - valid.HistorySize
			}
			kept = copy(newRecords, oldPage[start:])
			d.next = kept % valid.HistorySize
		} else {
			d.next = 0
		}
		d.filled = valid.HistorySize > 0 && kept == valid.HistorySize
		if d.filled {
			d.next = 0
		}
		d.records = newRecords
	}
	d.mu.Unlock()
	return nil
}

// Config returns a copy of the current configuration.
func (d *Dispatcher) Config() DispatchConfig {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.cfg
}

func (d *Dispatcher) configSnapshot() DispatchConfig {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.cfg
}

// ResetObservability clears records and counters without changing configuration.
func (d *Dispatcher) ResetObservability() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.records = make([]Decision, d.cfg.HistorySize)
	d.next = 0
	d.filled = false
	d.total = 0
	d.counts = DispatchCounts{
		ByRoute:          make(map[Route]int64),
		ByPrecision:      make(map[Precision]int64),
		ByRoutePrecision: make(map[RoutePrecisionKey]RoutePrecisionCounts),
	}
}

// ResetDispatchObservability resets the default dispatcher.
func ResetDispatchObservability() { defaultDispatcher.ResetObservability() }

func cloneCounts(c DispatchCounts) DispatchCounts {
	out := DispatchCounts{
		Calls:            c.Calls,
		Degraded:         c.Degraded,
		Failures:         c.Failures,
		ByRoute:          make(map[Route]int64, len(c.ByRoute)),
		ByPrecision:      make(map[Precision]int64, len(c.ByPrecision)),
		ByRoutePrecision: make(map[RoutePrecisionKey]RoutePrecisionCounts, len(c.ByRoutePrecision)),
	}
	for k, v := range c.ByRoute {
		out.ByRoute[k] = v
	}
	for k, v := range c.ByPrecision {
		out.ByPrecision[k] = v
	}
	for k, v := range c.ByRoutePrecision {
		out.ByRoutePrecision[k] = v
	}
	return out
}

// Counts returns a consistent copy of the counters.
func (d *Dispatcher) Counts() DispatchCounts {
	d.mu.Lock()
	defer d.mu.Unlock()
	return cloneCounts(d.counts)
}

// DispatchCounts returns a consistent copy of the default counters.
func DispatchCountsSnapshot() DispatchCounts { return defaultDispatcher.Counts() }

func (d *Dispatcher) historyLocked(limit int) []Decision {
	if d.cfg.HistorySize <= 0 || len(d.records) == 0 {
		return []Decision{}
	}
	var out []Decision
	if d.filled {
		out = make([]Decision, len(d.records))
		n := copy(out, d.records[d.next:])
		copy(out[n:], d.records[:d.next])
	} else {
		out = append([]Decision(nil), d.records[:d.next]...)
	}
	if limit > 0 && limit < len(out) {
		out = append([]Decision(nil), out[len(out)-limit:]...)
	}
	return out
}

// History returns retained decisions in chronological order. If limit is
// positive, at most the newest limit records are returned.
func (d *Dispatcher) History(limit int) []Decision {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.historyLocked(limit)
}

// DispatchHistory returns default-dispatcher decisions.
func DispatchHistory(limit int) []Decision { return defaultDispatcher.History(limit) }

// HistoryPage returns records plus the total accepted decision count and ring
// capacity.
func (d *Dispatcher) HistoryPage(limit int) HistoryPage {
	d.mu.Lock()
	defer d.mu.Unlock()
	recs := d.historyLocked(limit)
	return HistoryPage{Records: recs, Total: d.total, Capacity: d.cfg.HistorySize}
}

// DispatchHistoryPage queries the default dispatcher.
func DispatchHistoryPage(limit int) HistoryPage { return defaultDispatcher.HistoryPage(limit) }

// commit records one accepted dispatch. Invalid arguments must not reach
// commit; unsupported-but-valid operations may commit with err != nil.
func (d *Dispatcher) commit(dec Decision, callErr error) Decision {
	if callErr != nil {
		dec.OK = false
		dec.Error = callErr.Error()
	} else {
		dec.OK = true
	}
	if dec.At.IsZero() {
		dec.At = time.Now()
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	d.total++
	dec.ID = d.total
	d.counts.Calls++
	if callErr != nil {
		d.counts.Failures++
	}
	if dec.Degraded || dec.FallbackReason != "" {
		dec.Degraded = true
		d.counts.Degraded++
	}
	d.counts.ByRoute[dec.Route]++
	d.counts.ByPrecision[dec.Precision]++
	key := RoutePrecisionKey{Route: dec.Route, Precision: dec.Precision}
	rpc := d.counts.ByRoutePrecision[key]
	rpc.Calls++
	if callErr != nil {
		rpc.Failures++
	}
	if dec.Degraded {
		rpc.Degraded++
	}
	d.counts.ByRoutePrecision[key] = rpc

	if d.cfg.Collect && d.cfg.HistorySize > 0 {
		if len(d.records) != d.cfg.HistorySize {
			d.records = make([]Decision, d.cfg.HistorySize)
		}
		d.records[d.next] = dec
		d.next++
		if d.next == d.cfg.HistorySize {
			d.next = 0
			d.filled = true
		}
	}
	return dec
}

type callOptions struct {
	precision    Precision
	hasPrecision bool
}

// CallOption configures one dispatched call.
type CallOption func(*callOptions)

// WithPrecision overrides the configured default precision for one call.
func WithPrecision(p Precision) CallOption {
	return func(o *callOptions) {
		o.precision = p
		o.hasPrecision = true
	}
}

func resolvePrecision(cfg DispatchConfig, opts []CallOption) (Precision, error) {
	o := callOptions{precision: cfg.DefaultPrecision}
	for _, fn := range opts {
		fn(&o)
	}
	if o.hasPrecision && o.precision != PrecisionStandard && o.precision != PrecisionHigh {
		return o.precision, fmt.Errorf("mat: unknown dispatch precision %s", o.precision)
	}
	return o.precision, nil
}

type workspacePool[T Numeric] struct {
	mu   sync.Mutex
	free [][]T
}

func (p *workspacePool[T]) get(n int) []T {
	b, _ := p.getE(n, nil)
	return b
}

func (p *workspacePool[T]) getE(n int, allocErr error) ([]T, error) {
	if n <= 0 {
		return nil, nil
	}
	if allocErr != nil {
		return nil, allocErr
	}
	p.mu.Lock()
	for i, b := range p.free {
		if cap(b) >= n {
			p.free = append(p.free[:i], p.free[i+1:]...)
			p.mu.Unlock()
			return b[:n], nil
		}
	}
	p.mu.Unlock()
	return make([]T, n), nil
}

func (p *workspacePool[T]) put(b []T) {
	if b == nil {
		return
	}
	p.mu.Lock()
	p.free = append(p.free, b[:cap(b)])
	p.mu.Unlock()
}

type exactWorkspacePool struct {
	mu   sync.Mutex
	free [][]exactAccumulator
}

func (p *exactWorkspacePool) get(n int) []exactAccumulator {
	if n <= 0 {
		return nil
	}
	p.mu.Lock()
	for i, b := range p.free {
		if cap(b) >= n {
			p.free = append(p.free[:i], p.free[i+1:]...)
			p.mu.Unlock()
			return b[:n]
		}
	}
	p.mu.Unlock()
	return make([]exactAccumulator, n)
}

func (p *exactWorkspacePool) put(b []exactAccumulator) {
	if b == nil {
		return
	}
	p.mu.Lock()
	p.free = append(p.free, b[:cap(b)])
	p.mu.Unlock()
}

func workspacePoolFor[T Numeric](d *Dispatcher) *workspacePool[T] {
	var zero T
	switch any(zero).(type) {
	case float64:
		return any(&d.pool64).(*workspacePool[T])
	case float32:
		return any(&d.pool32).(*workspacePool[T])
	default:
		panic("mat: unsupported dispatch numeric type")
	}
}
