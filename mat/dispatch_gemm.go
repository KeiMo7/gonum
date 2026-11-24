// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"sync"
)

// Numeric is the set of scalar precisions supported by the unified dispatcher.
type Numeric interface {
	~float32 | ~float64
}

// RawMatrixN is a stride-aware generic matrix descriptor.
//
// Rows and Cols describe the physical row-major storage. When Trans is true,
// the descriptor presents the logical transpose, so its logical dimensions are
// Cols by Rows. Stride is the distance between physical rows and defaults to
// Cols when zero.
type RawMatrixN[T Numeric] struct {
	Rows   int
	Cols   int
	Stride int
	Data   []T
	Trans  bool
}

// RawMatrix64 is the float64 specialization.
type RawMatrix64 = RawMatrixN[float64]

// RawMatrix32 is the float32 specialization.
type RawMatrix32 = RawMatrixN[float32]

// RawFromDense returns a raw descriptor aliasing d's storage.
func RawFromDense(d *Dense) RawMatrix64 {
	if d == nil {
		return RawMatrix64{}
	}
	return RawMatrix64{
		Rows:   d.mat.Rows,
		Cols:   d.mat.Cols,
		Stride: d.mat.Stride,
		Data:   d.mat.Data,
	}
}

func (m RawMatrixN[T]) normalized(name string) (RawMatrixN[T], error) {
	if m.Rows < 0 || m.Cols < 0 {
		return m, fmt.Errorf("%w: %s has negative dimensions %d×%d", ErrShape, name, m.Rows, m.Cols)
	}
	if m.Cols == 0 && m.Stride == 0 {
		m.Stride = 0
	} else if m.Stride == 0 {
		m.Stride = m.Cols
	}
	if m.Cols > 0 && m.Stride < m.Cols {
		return m, fmt.Errorf("%w: %s stride %d is less than columns %d", ErrShape, name, m.Stride, m.Cols)
	}
	if m.Rows > 0 {
		need := (m.Rows-1)*m.Stride + m.Cols
		if need > len(m.Data) {
			return m, fmt.Errorf("%w: %s storage length %d is too short; need %d", ErrShape, name, len(m.Data), need)
		}
	}
	return m, nil
}

func (m RawMatrixN[T]) logicalDims() (int, int) {
	if m.Trans {
		return m.Cols, m.Rows
	}
	return m.Rows, m.Cols
}

func (m RawMatrixN[T]) at(i, j int) T {
	if m.Trans {
		return m.Data[j*m.Stride+i]
	}
	return m.Data[i*m.Stride+j]
}

func (m RawMatrixN[T]) set(i, j int, v T) {
	if m.Trans {
		m.Data[j*m.Stride+i] = v
		return
	}
	m.Data[i*m.Stride+j] = v
}

func numericBytes[T Numeric]() int {
	var zero T
	switch any(zero).(type) {
	case float32:
		return 4
	case float64:
		return 8
	default:
		return int(^uint(0) >> 3)
	}
}

type gemmMode uint8

const (
	gemmSequential gemmMode = iota
	gemmOutputTiles
	gemmKPartitions
)

type gemmPlan struct {
	dec     Decision
	mode    gemmMode
	tilesM  int
	tilesN  int
	tiles   int
	kParts  int
	wsElems int
	useWS   bool
}

func ceilDiv(a, b int) int {
	if a == 0 || b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

func chooseChunk(requested, dim, fallback int, adaptive bool) (size int, automatic bool) {
	if dim == 0 {
		return 0, adaptive
	}
	base := requested
	if base <= 0 {
		base = fallback
	}
	return minInt(base, dim), adaptive
}

func chunkPanelElements(cm, cn, ck int) int64 {
	return int64(cm)*int64(ck) + int64(cn)*int64(ck) + int64(cm)*int64(cn)
}

func adaptChunksToBudget(budget int64, elemSize, cm, cn, ck int, autoM, autoN, autoK bool) (int, int, int) {
	if budget <= 0 || elemSize <= 0 {
		return cm, cn, ck
	}
	shrink := func(v int) int {
		if v <= 1 {
			return 1
		}
		nv := v / 2
		if nv < 1 {
			nv = 1
		}
		return nv
	}
	for chunkPanelElements(cm, cn, ck)*int64(elemSize) > budget {
		changed := false
		if autoK && ck > 1 {
			ck = shrink(ck)
			changed = true
		}
		if chunkPanelElements(cm, cn, ck)*int64(elemSize) <= budget {
			break
		}
		if autoM && cm > 1 {
			cm = shrink(cm)
			changed = true
		}
		if chunkPanelElements(cm, cn, ck)*int64(elemSize) <= budget {
			break
		}
		if autoN && cn > 1 {
			cn = shrink(cn)
			changed = true
		}
		if !changed {
			break
		}
	}
	return cm, cn, ck
}

func planGemm(cfg DispatchConfig, p Precision, m, n, k int, ta, tb bool, elemSize int) gemmPlan {
	cm, autoM := chooseChunk(cfg.ChunkM, m, defaultChunkM, cfg.AdaptiveChunks)
	cn, autoN := chooseChunk(cfg.ChunkN, n, defaultChunkN, cfg.AdaptiveChunks)
	ck, autoK := chooseChunk(cfg.ChunkK, k, defaultChunkK, cfg.AdaptiveChunks)
	cm, cn, ck = adaptChunksToBudget(cfg.MemoryBudget, elemSize, cm, cn, ck, autoM, autoN, autoK)

	kParts := 0
	if k > 0 {
		kParts = ceilDiv(k, deterministicKPartition)
	}
	tm := ceilDiv(m, max1(cm))
	tn := ceilDiv(n, max1(cn))
	if m == 0 {
		tm = 0
	}
	if n == 0 {
		tn = 0
	}
	tiles := tm * tn

	work := int64(m) * int64(n) * int64(max1(k)) * 2
	work += int64(m*n) * 4
	dec := Decision{
		Op:           "gemm",
		Route:        RouteGEMMDirect,
		Precision:    p,
		M:            m,
		N:            n,
		K:            k,
		TransA:       ta,
		TransB:       tb,
		Chunk:        Chunk{M: cm, N: cn, K: ck},
		Workers:      1,
		Goroutines:   0,
		KPartitions:  max1(kParts),
		WorkUnits:    work,
		MemoryBudget: cfg.MemoryBudget,
		Accumulator:  accumulatorName(p),
		Specialized:  false,
	}

	plan := gemmPlan{dec: dec, tilesM: tm, tilesN: tn, tiles: tiles, kParts: max1(kParts)}
	if m == 0 || n == 0 {
		plan.dec.KPartitions = 0
		return plan
	}

	maxW := cfg.MaxWorkers
	if work >= parallelWorkThreshold && maxW > 1 {
		switch {
		case tiles >= 2:
			plan.mode = gemmOutputTiles
			plan.dec.Workers = minInt(maxW, tiles)
		case kParts >= 2:
			tasks := m * n * kParts
			plan.mode = gemmKPartitions
			plan.dec.Workers = minInt(maxW, tasks)
		case m*n >= 2:
			// A single large tile still contains independent output cells.
			plan.mode = gemmOutputTiles
			plan.tiles = m * n
			plan.dec.Workers = minInt(maxW, m*n)
		}
		if plan.dec.Workers <= 1 {
			plan.mode = gemmSequential
		} else {
			plan.dec.Goroutines = plan.dec.Workers - 1
		}
	}

	if plan.mode == gemmKPartitions {
		plan.wsElems = m * n * kParts
		wantBytes := int64(plan.wsElems) * int64(numericBytes[float64]()) // overwritten by caller with T size
		_ = wantBytes
	}
	return plan
}

func accumulatorName(p Precision) string {
	if p == PrecisionHigh {
		return "exact-superaccumulator"
	}
	var zero float32
	_ = zero
	return "native-element"
}

func max1(x int) int {
	if x <= 0 {
		return 1
	}
	return x
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// GemmOn performs a deterministic general matrix multiplication using d.
//
// Storage is row-major. RawMatrixN.Rows/Cols describe the physical matrix and
// Stride is the distance between physical rows; Trans presents its logical
// transpose. The reduction dimension is always traversed with k increasing.
// Parallel K work is merged in ascending K-partition order, and M/N work owns
// disjoint outputs, so worker count and scheduling do not change the bits.
//
// A nil Dispatcher uses DefaultDispatch.
func GemmOn[T Numeric](d *Dispatcher, c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T, opts ...CallOption) (Decision, error) {
	if d == nil {
		d = defaultDispatcher
	}
	cfg := d.configSnapshot()
	p, err := resolvePrecision(cfg, opts)
	if err != nil {
		return Decision{}, err
	}

	c, err = c.normalized("c")
	if err != nil {
		return Decision{}, err
	}
	a, err = a.normalized("a")
	if err != nil {
		return Decision{}, err
	}
	b, err = b.normalized("b")
	if err != nil {
		return Decision{}, err
	}

	ar, ak := a.logicalDims()
	bk, bn := b.logicalDims()
	cm, cn := c.logicalDims()
	if ak != bk {
		err := fmt.Errorf("%w: A columns %d != B rows %d", ErrShape, ak, bk)
		return Decision{}, err
	}
	if cm != ar || cn != bn {
		err := fmt.Errorf("%w: C is %d×%d but product is %d×%d", ErrShape, cm, cn, ar, bn)
		return Decision{}, err
	}

	size := numericBytes[T]()
	plan := planGemm(cfg, p, ar, bn, ak, a.Trans, b.Trans, size)
	var zeroT T
	if p == PrecisionStandard {
		plan.dec.Accumulator = fmt.Sprintf("native-%T", zeroT)
	} else {
		plan.dec.Accumulator = fmt.Sprintf("exact-superaccumulator-%T", zeroT)
	}

	// Workspace planning and budget degradation. A zero budget is valid and
	// means "no extra workspace"; it takes the same deterministic direct path.
	if plan.mode == gemmKPartitions {
		want := int64(plan.wsElems) * int64(size)
		if p == PrecisionHigh {
			// Conservative bound for the exact super-accumulator state used
			// by each parallel K partition.
			want += int64(plan.wsElems) * exactAccumulatorEstimateBytes
		}
		if cfg.MemoryBudget < want {
			plan.dec.Route = RouteGEMMDirect
			plan.dec.Degraded = true
			plan.dec.FallbackReason = FallbackMemoryBudget
			plan.dec.WorkspaceBytes = 0
			plan.dec.Workers = 1
			plan.dec.Goroutines = 0
			plan.mode = gemmSequential
			plan.useWS = false
		} else {
			plan.dec.Route = RouteGEMMWorkspace
			plan.dec.WorkspaceBytes = want
			plan.useWS = true
		}
	}

	dec, runErr := runGemm(d, cfg, plan, c, a, b, alpha, beta)
	dec = d.commit(dec, runErr)
	return dec, runErr
}

// GemmN performs GEMM on the default dispatcher.
func GemmN[T Numeric](c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T, opts ...CallOption) (Decision, error) {
	return GemmOn(defaultDispatcher, c, a, b, alpha, beta, opts...)
}

// MulN computes C = A*B on the default dispatcher.
func MulN[T Numeric](c RawMatrixN[T], a, b RawMatrixN[T], opts ...CallOption) (Decision, error) {
	var one, zero T
	switch any(one).(type) {
	case float32:
		one = any(float32(1)).(T)
		zero = any(float32(0)).(T)
	case float64:
		one = any(float64(1)).(T)
		zero = any(float64(0)).(T)
	}
	return GemmN(c, a, b, one, zero, opts...)
}

func isZeroT[T Numeric](x T) bool { return x == 0 }
func isOneT[T Numeric](x T) bool  { return x == 1 }

func runGemm[T Numeric](d *Dispatcher, cfg DispatchConfig, plan gemmPlan, c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T) (Decision, error) {
	dec := plan.dec
	m, n, k := dec.M, dec.N, dec.K
	if m == 0 || n == 0 {
		return dec, nil
	}

	// alpha == 0: C = beta*C. Do not read A/B products.
	if isZeroT(alpha) {
		scaleOutput(c, m, n, beta)
		return dec, nil
	}

	// k == 0 has an empty dot product, hence C = alpha*0 + beta*C.
	if k == 0 {
		scaleOutput(c, m, n, beta)
		return dec, nil
	}

	switch plan.mode {
	case gemmKPartitions:
		if plan.useWS {
			if dec.Precision == PrecisionHigh {
				if d.workspaceAllocErr != nil {
					dec.Route = RouteGEMMDirect
					dec.Degraded = true
					dec.FallbackReason = FallbackWorkspaceAllocation
					dec.WorkspaceBytes = 0
					dec.Workers = 1
					dec.Goroutines = 0
					runKSequential(m, n, k, plan.kParts, c, a, b, alpha, beta, cfg.ChunkK, dec.Precision)
					return dec, nil
				}
				runKParallelExact(d, dec.Workers, m, n, k, plan.kParts, c, a, b, alpha, beta)
				return dec, nil
			}
			pool := workspacePoolFor[T](d)
			ws, wsErr := pool.getE(plan.wsElems, d.workspaceAllocErr)
			if wsErr != nil {
				dec.Route = RouteGEMMDirect
				dec.Degraded = true
				dec.FallbackReason = FallbackWorkspaceAllocation
				dec.WorkspaceBytes = 0
				dec.Workers = 1
				dec.Goroutines = 0
				runKSequential(m, n, k, plan.kParts, c, a, b, alpha, beta, cfg.ChunkK, dec.Precision)
				return dec, nil
			}
			runKParallel(dec.Workers, m, n, k, plan.kParts, c, a, b, alpha, beta, cfg.ChunkK, dec.Precision, ws)
			pool.put(ws)
			return dec, nil
		}
		runKSequential(m, n, k, plan.kParts, c, a, b, alpha, beta, cfg.ChunkK, dec.Precision)
		return dec, nil

	case gemmOutputTiles:
		runOutputTiles(plan, c, a, b, alpha, beta)
		return dec, nil

	default:
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				v := reduceDot(a, b, i, j, k, plan.kParts, cfg.ChunkK, dec.Precision)
				writeGemmCell(c, i, j, alpha, beta, v)
			}
		}
		return dec, nil
	}
}

func scaleOutput[T Numeric](c RawMatrixN[T], m, n int, beta T) {
	if isOneT(beta) {
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if isZeroT(beta) {
				c.set(i, j, 0)
			} else {
				c.set(i, j, beta*c.at(i, j))
			}
		}
	}
}

func writeGemmCell[T Numeric](c RawMatrixN[T], i, j int, alpha, beta, dot T) {
	var v T
	if isOneT(alpha) {
		v = dot
	} else {
		v = alpha * dot
	}
	if isZeroT(beta) {
		c.set(i, j, v)
	} else if isOneT(beta) {
		c.set(i, j, v+c.at(i, j))
	} else {
		c.set(i, j, v+beta*c.at(i, j))
	}
}

func segmentBounds(k, s int) (int, int) {
	start := s * deterministicKPartition
	end := start + deterministicKPartition
	if end > k {
		end = k
	}
	return start, end
}

func reduceDot[T Numeric](a, b RawMatrixN[T], row, col, k, parts, chunkK int, p Precision) T {
	if p == PrecisionHigh {
		// Exact accumulation is associative, so running it over the full K
		// range gives the same bits as merging independently exact K parts.
		var exact exactAccumulator
		exact.reset()
		if exactSegmentInto(&exact, a, b, row, col, 0, k) {
			return exactValueT[T](&exact)
		}
		// Non-finite IEEE values require the hardware propagation path.
		return reduceDot(a, b, row, col, k, parts, chunkK, PrecisionStandard)
	}

	var total T
	for s := 0; s < parts; s++ {
		st, en := segmentBounds(k, s)
		partial := dotSegment(a, b, row, col, st, en, chunkK, PrecisionStandard)
		total += partial
	}
	return total
}

func dotSegment[T Numeric](a, b RawMatrixN[T], row, col, start, end, chunkK int, p Precision) T {
	if p == PrecisionHigh {
		if v, ok := exactSegment[T](a, b, row, col, start, end); ok {
			return v
		}
	}
	if chunkK <= 0 || chunkK > end-start {
		chunkK = end - start
	}
	var partial T
	for ck := start; ck < end; ck += chunkK {
		ce := ck + chunkK
		if ce > end {
			ce = end
		}
		for kk := ck; kk < ce; kk++ {
			partial += a.at(row, kk) * b.at(kk, col)
		}
	}
	return partial
}

func runKSequential[T Numeric](m, n, k, parts int, c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T, chunkK int, p Precision) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			total := reduceDot(a, b, i, j, k, parts, chunkK, p)
			writeGemmCell(c, i, j, alpha, beta, total)
		}
	}
}

func runKParallel[T Numeric](workers, m, n, k, parts int, c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T, chunkK int, p Precision, ws []T) {
	tasks := m * n * parts
	var wg sync.WaitGroup
	next := make(chan int, tasks)
	for task := 0; task < tasks; task++ {
		next <- task
	}
	close(next)

	work := func() {
		for task := range next {
			cell := task / parts
			s := task % parts
			i := cell / n
			j := cell % n
			st, en := segmentBounds(k, s)
			ws[task] = dotSegment(a, b, i, j, st, en, chunkK, p)
		}
	}
	for w := 1; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work()
		}()
	}
	// The calling goroutine is the final execution lane.
	work()
	wg.Wait()

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			cell := i*n + j
			total := combinePartials(ws[cell*parts:(cell+1)*parts], p)
			writeGemmCell(c, i, j, alpha, beta, total)
		}
	}
}

func runKParallelExact[T Numeric](d *Dispatcher, workers, m, n, k, parts int, c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T) {
	tasks := m * n * parts
	accs := d.exact.get(tasks)
	defer d.exact.put(accs)

	next := make(chan int, tasks)
	for task := 0; task < tasks; task++ {
		next <- task
	}
	close(next)
	var wg sync.WaitGroup
	work := func() {
		for task := range next {
			cell := task / parts
			s := task % parts
			i := cell / n
			j := cell % n
			st, en := segmentBounds(k, s)
			acc := &accs[task]
			acc.reset()
			exactSegmentInto(acc, a, b, i, j, st, en)
		}
	}
	for w := 1; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			cell := i*n + j
			var merged exactAccumulator
			merged.reset()
			hardware := false
			for s := 0; s < parts; s++ {
				src := &accs[cell*parts+s]
				if !src.finite {
					hardware = true
					break
				}
				mergeExactAccumulator(&merged, src)
			}
			var total T
			if hardware {
				total = reduceDot(a, b, i, j, k, parts, deterministicKPartition, PrecisionStandard)
			} else {
				total = exactValueT[T](&merged)
			}
			writeGemmCell(c, i, j, alpha, beta, total)
		}
	}
}

func runOutputTiles[T Numeric](plan gemmPlan, c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T) {
	workers := plan.dec.Workers
	m, n := plan.dec.M, plan.dec.N
	if workers <= 1 {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				v := reduceDot(a, b, i, j, plan.dec.K, plan.kParts, plan.dec.Chunk.K, plan.dec.Precision)
				writeGemmCell(c, i, j, alpha, beta, v)
			}
		}
		return
	}

	if plan.tiles < 2 {
		// One M/N tile can still contain many independent output cells.
		runIndexedJobs(workers, m*n, func(cell int) {
			i := cell / n
			j := cell % n
			v := reduceDot(a, b, i, j, plan.dec.K, plan.kParts, plan.dec.Chunk.K, plan.dec.Precision)
			writeGemmCell(c, i, j, alpha, beta, v)
		})
		return
	}

	type tile struct{ ti, tj int }
	tasks := make(chan tile, plan.tiles)
	for ti := 0; ti < plan.tilesM; ti++ {
		for tj := 0; tj < plan.tilesN; tj++ {
			tasks <- tile{ti, tj}
		}
	}
	close(tasks)
	var wg sync.WaitGroup
	plan.dec.Goroutines = workers - 1
	for w := 1; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for tl := range tasks {
				executeTile(plan, tl.ti, tl.tj, c, a, b, alpha, beta)
			}
		}()
	}
	for tl := range tasks {
		executeTile(plan, tl.ti, tl.tj, c, a, b, alpha, beta)
	}
	wg.Wait()
}

func executeTile[T Numeric](plan gemmPlan, ti, tj int, c RawMatrixN[T], a, b RawMatrixN[T], alpha, beta T) {
	m, n, k := plan.dec.M, plan.dec.N, plan.dec.K
	cm, cn := plan.dec.Chunk.M, plan.dec.Chunk.N
	i0 := ti * cm
	i1 := minInt(i0+cm, m)
	j0 := tj * cn
	j1 := minInt(j0+cn, n)
	for i := i0; i < i1; i++ {
		for j := j0; j < j1; j++ {
			v := reduceDot(a, b, i, j, k, plan.kParts, plan.dec.Chunk.K, plan.dec.Precision)
			writeGemmCell(c, i, j, alpha, beta, v)
		}
	}
}
