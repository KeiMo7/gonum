// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
)

func dispatchValues[T Numeric](seed int) func(i, j int) T {
	return func(i, j int) T {
		x := math.Sin(float64((i+1)*17+(j+1)*31+seed*7) * 0.137)
		switch any(T(0)).(type) {
		case float32:
			return any(float32(x)).(T)
		default:
			return any(x).(T)
		}
	}
}

func dispatchDense[T Numeric](r, c, pad, seed int) RawMatrixN[T] {
	stride := c + pad
	data := make([]T, r*stride)
	fn := dispatchValues[T](seed)
	m := RawMatrixN[T]{Rows: r, Cols: c, Stride: stride, Data: data}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*stride+j] = fn(i, j)
		}
	}
	return m
}

func dispatchTransposeOf[T Numeric](in RawMatrixN[T], pad int) RawMatrixN[T] {
	r, c := in.Rows, in.Cols
	stride := r + pad
	out := RawMatrixN[T]{Rows: c, Cols: r, Stride: stride, Data: make([]T, c*stride), Trans: true}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Data[j*stride+i] = in.Data[i*in.Stride+j]
		}
	}
	return out
}

func dispatchNaive[T Numeric](m, n, k int, a, b RawMatrixN[T], alpha, beta T, initial []T) []T {
	out := append([]T(nil), initial...)
	if out == nil {
		out = make([]T, m*n)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s T
			for q := 0; q < k; q++ {
				s += a.at(i, q) * b.at(q, j)
			}
			idx := i*n + j
			if isZeroT(alpha) {
				switch {
				case isZeroT(beta):
					out[idx] = 0
				case isOneT(beta):
					// Keep initial value.
				default:
					out[idx] = beta * out[idx]
				}
				continue
			}
			if isZeroT(beta) {
				out[idx] = alpha * s
			} else {
				out[idx] = alpha*s + beta*out[idx]
			}
		}
	}
	return out
}

func dispatchBitsEqual[T Numeric](x, y T) bool {
	switch a := any(x).(type) {
	case float32:
		b := any(y).(float32)
		return math.Float32bits(a) == math.Float32bits(b)
	case float64:
		b := any(y).(float64)
		return math.Float64bits(a) == math.Float64bits(b)
	default:
		return x == y
	}
}

func dispatchRequireMatrixBits[T Numeric](t *testing.T, label string, got RawMatrixN[T], want []T) {
	t.Helper()
	m, n := got.logicalDims()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			w := want[i*n+j]
			g := got.at(i, j)
			if !dispatchBitsEqual(g, w) {
				t.Fatalf("%s: element (%d,%d) bits differ: got %v want %v", label, i, j, g, w)
			}
		}
	}
}

func dispatchApproxEqual(t *testing.T, label string, got, want float64, tol float64) {
	t.Helper()
	diff := math.Abs(got - want)
	scale := math.Max(1, math.Abs(want))
	if diff > tol*scale && diff > tol {
		t.Fatalf("%s: got %v want %v diff %g tol %g", label, got, want, diff, tol)
	}
}

func dispatchTestConfig(workers int) DispatchConfig {
	cfg := DefaultDispatchConfig()
	cfg.ChunkM = 7
	cfg.ChunkN = 5
	cfg.ChunkK = 11
	cfg.AdaptiveChunks = false
	cfg.MaxWorkers = workers
	cfg.MemoryBudget = 1 << 20
	cfg.HistorySize = 32
	return cfg
}

func TestDispatchConfigValidation(t *testing.T) {
	base := DefaultDispatchConfig()
	tests := []struct {
		name      string
		modify    func(*DispatchConfig)
		wantField string
	}{
		{"chunk-m-zero", func(c *DispatchConfig) { c.ChunkM = 0 }, "ChunkM"},
		{"chunk-m-negative", func(c *DispatchConfig) { c.ChunkM = -2 }, "ChunkM"},
		{"chunk-n-zero", func(c *DispatchConfig) { c.ChunkN = 0 }, "ChunkN"},
		{"chunk-k-zero", func(c *DispatchConfig) { c.ChunkK = 0 }, "ChunkK"},
		{"workers-zero", func(c *DispatchConfig) { c.MaxWorkers = 0 }, "MaxWorkers"},
		{"workers-negative", func(c *DispatchConfig) { c.MaxWorkers = -2 }, "MaxWorkers"},
		{"budget-negative", func(c *DispatchConfig) { c.MemoryBudget = -1 }, "MemoryBudget"},
		{"history-negative", func(c *DispatchConfig) { c.HistorySize = -1 }, "HistorySize"},
		{"precision-unknown", func(c *DispatchConfig) { c.DefaultPrecision = Precision(99) }, "DefaultPrecision"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := base
			test.modify(&cfg)
			_, err := NewDispatcher(cfg)
			if err == nil {
				t.Fatal("expected configuration error")
			}
			if !containsString(err.Error(), test.wantField) {
				t.Fatalf("error %q does not mention %q", err.Error(), test.wantField)
			}

			d := MustNewDispatcher(DefaultDispatchConfig())
			old := d.Config()
			if err := d.Configure(cfg); err == nil {
				t.Fatal("expected Configure error")
			}
			if got := d.Config(); got != old {
				t.Fatal("invalid configuration changed active config")
			}
		})
	}
}

func containsString(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}

func TestDispatchShapeMatrixAndTransposes(t *testing.T) {
	shapes := [][3]int{
		{0, 4, 5},
		{3, 0, 5},
		{3, 4, 0},
		{1, 1, 1},
		{1, 1, 9},
		{2, 3, 5},
		{7, 5, 13},
		{17, 19, 23},
		{33, 17, 1025},
	}
	for _, shape := range shapes {
		m, n, k := shape[0], shape[1], shape[2]
		for ta := 0; ta < 2; ta++ {
			for tb := 0; tb < 2; tb++ {
				name := fmt.Sprintf("%dx%dx%d/ta=%d/tb=%d", m, n, k, ta, tb)
				t.Run(name, func(t *testing.T) {
					testDispatchOneShape[float64](t, m, n, k, ta == 1, tb == 1)
					testDispatchOneShape[float32](t, m, n, k, ta == 1, tb == 1)
				})
			}
		}
	}
}

func testDispatchOneShape[T Numeric](t *testing.T, m, n, k int, ta, tb bool) {
	t.Helper()
	d := MustNewDispatcher(dispatchTestConfig(4))
	rawA := dispatchDense[T](m, k, 2, 1)
	rawB := dispatchDense[T](k, n, 3, 2)
	a := rawA
	if ta {
		a = dispatchTransposeOf(rawA, 1)
	}
	b := rawB
	if tb {
		b = dispatchTransposeOf(rawB, 2)
	}
	c := RawMatrixN[T]{Rows: m, Cols: n, Stride: n, Data: make([]T, m*max1(n))}
	var alpha, beta T = 1, 0
	dec, err := GemmOn(d, c, a, b, alpha, beta)
	if err != nil {
		t.Fatalf("dispatch failed: %v", err)
	}
	if dec.M != m || dec.N != n || dec.K != k {
		t.Fatalf("decision dims = %d,%d,%d want %d,%d,%d", dec.M, dec.N, dec.K, m, n, k)
	}
	if dec.TransA != ta || dec.TransB != tb {
		t.Fatalf("transpose flags = %v,%v want %v,%v", dec.TransA, dec.TransB, ta, tb)
	}
	wantCM := minInt(7, m)
	wantCN := minInt(5, n)
	wantCK := minInt(11, k)
	if dec.Chunk.M != wantCM || dec.Chunk.N != wantCN || dec.Chunk.K != wantCK {
		t.Fatalf("chunks = %+v want %d,%d,%d", dec.Chunk, wantCM, wantCN, wantCK)
	}
	if dec.Workers < 1 || dec.Workers > 4 {
		t.Fatalf("workers %d outside [1,4]", dec.Workers)
	}
	if dec.Goroutines != dec.Workers-1 {
		t.Fatalf("helper goroutines %d want workers-1 %d", dec.Goroutines, dec.Workers-1)
	}
	if int64(dec.Workers) > int64(4) {
		t.Fatal("concurrency exceeds configured limit")
	}
	if m > 0 && n > 0 {
		wantParts := ceilDiv(k, deterministicKPartition)
		if wantParts < 1 {
			wantParts = 1
		}
		if dec.KPartitions != wantParts {
			t.Fatalf("K partitions %d want %d", dec.KPartitions, wantParts)
		}
	}
	if int64(m)*int64(n)*int64(k) <= 512 && dec.Workers != 1 {
		t.Fatalf("small job workers=%d, want direct caller", dec.Workers)
	}

	ref := dispatchNaive(m, n, k, a, b, alpha, beta, nil)
	tol := 1e-10
	if _, is32 := any(T(0)).(float32); is32 {
		tol = 2e-4
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			got := float64(c.at(i, j))
			want := float64(ref[i*n+j])
			dispatchApproxEqual(t, "gemm", got, want, tol)
		}
	}
}

func TestDispatchLongKPlansParallel(t *testing.T) {
	cfg := dispatchTestConfig(4)
	d := MustNewDispatcher(cfg)
	a := dispatchDense[float32](1, 20000, 0, 7)
	b := dispatchDense[float32](20000, 1, 0, 8)
	c := RawMatrix32{Rows: 1, Cols: 1, Stride: 1, Data: make([]float32, 1)}
	dec, err := GemmOn(d, c, a, b, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if dec.KPartitions < 2 {
		t.Fatalf("KPartitions=%d want >1", dec.KPartitions)
	}
	if dec.Workers <= 1 {
		t.Fatalf("Workers=%d want >1 for long K narrow output", dec.Workers)
	}
	if dec.Goroutines < 1 {
		t.Fatalf("Goroutines=%d want helpers", dec.Goroutines)
	}
	if dec.Route != RouteGEMMWorkspace {
		t.Fatalf("route=%s want workspace with budget", dec.Route)
	}
	if dec.WorkspaceBytes <= 0 {
		t.Fatal("workspace bytes not recorded")
	}
}

func TestDispatchEquivalentLayoutsAndLimitsBitwise(t *testing.T) {
	const m, n, k = 19, 17, 29
	baseA := dispatchDense[float64](m, k, 0, 11)
	baseB := dispatchDense[float64](k, n, 0, 12)
	layouts := []struct {
		name string
		a    RawMatrix64
		b    RawMatrix64
	}{
		{"direct", baseA, baseB},
		{"a-transposed-storage", dispatchTransposeOf(baseA, 4), baseB},
		{"b-transposed-storage", baseA, dispatchTransposeOf(baseB, 5)},
		{"both-transposed-padded", dispatchTransposeOf(baseA, 2), dispatchTransposeOf(baseB, 3)},
	}

	var want []float64
	for idx, layout := range layouts {
		d := MustNewDispatcher(dispatchTestConfig(4))
		c := RawMatrix64{Rows: m, Cols: n, Stride: n + 4, Data: make([]float64, m*(n+4))}
		_, err := GemmOn(d, c, layout.a, layout.b, 1, 0)
		if err != nil {
			t.Fatalf("%s: %v", layout.name, err)
		}
		got := make([]float64, m*n)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				got[i*n+j] = c.at(i, j)
			}
		}
		if idx == 0 {
			want = got
		} else {
			dispatchRequireMatrixBits(t, layout.name, RawMatrix64{Rows: m, Cols: n, Stride: n, Data: got}, want)
		}
	}

	// Multiple worker caps must produce identical bits.
	for _, workers := range []int{1, 2, 3, 8} {
		cfg := dispatchTestConfig(workers)
		cfg.MemoryBudget = 1 << 20
		d := MustNewDispatcher(cfg)
		c := RawMatrix64{Rows: m, Cols: n, Stride: n, Data: make([]float64, m*n)}
		for repeat := 0; repeat < 5; repeat++ {
			if _, err := GemmOn(d, c, baseA, baseB, 1, 0); err != nil {
				t.Fatal(err)
			}
			dispatchRequireMatrixBits(t, fmt.Sprintf("workers-%d-repeat-%d", workers, repeat), c, want)
		}
	}
}

func TestDispatchChunkChangesBitwiseStable(t *testing.T) {
	const m, n, k = 37, 31, 43
	a := dispatchDense[float64](m, k, 1, 21)
	b := dispatchDense[float64](k, n, 2, 22)
	chunkSets := [][3]int{{1, 1, 1}, {3, 5, 7}, {8, 13, 17}, {64, 64, 256}}
	var want []float64
	for idx, chunks := range chunkSets {
		cfg := dispatchTestConfig(4)
		cfg.ChunkM, cfg.ChunkN, cfg.ChunkK = chunks[0], chunks[1], chunks[2]
		d := MustNewDispatcher(cfg)
		c := RawMatrix64{Rows: m, Cols: n, Stride: n, Data: make([]float64, m*n)}
		if _, err := GemmOn(d, c, a, b, 1, 0); err != nil {
			t.Fatal(err)
		}
		if idx == 0 {
			want = append([]float64(nil), c.Data...)
		} else {
			dispatchRequireMatrixBits(t, fmt.Sprintf("chunks-%v", chunks), c, want)
		}
	}

	// Long K exercises fixed reduction partitions and K parallelism.
	const kn = 5000
	ka := dispatchDense[float32](1, kn, 0, 31)
	kb := dispatchDense[float32](kn, 1, 0, 32)
	var want32 float32
	for idx, chunks := range chunkSets {
		cfg := dispatchTestConfig(3)
		cfg.ChunkM, cfg.ChunkN, cfg.ChunkK = chunks[0], chunks[1], chunks[2]
		d := MustNewDispatcher(cfg)
		c := RawMatrix32{Rows: 1, Cols: 1, Stride: 1, Data: make([]float32, 1)}
		if _, err := GemmOn(d, c, ka, kb, 1, 0); err != nil {
			t.Fatal(err)
		}
		if idx == 0 {
			want32 = c.Data[0]
		} else if math.Float32bits(c.Data[0]) != math.Float32bits(want32) {
			t.Fatalf("chunk set %v changed bits: %v vs %v", chunks, c.Data[0], want32)
		}
	}
}

func TestDispatchCancellationPrecisionTiers(t *testing.T) {
	// This vector has exact sum 96. Native float32 accumulation loses all 764
	// 1/8 terms and retains only the final 1/2 rounding step, yielding 0.5.
	const k32 = 767
	a32 := RawMatrix32{Rows: 1, Cols: k32, Stride: k32, Data: make([]float32, k32)}
	b32 := RawMatrix32{Rows: k32, Cols: 1, Stride: 1, Data: make([]float32, k32)}
	for i := range b32.Data {
		b32.Data[i] = 1
	}
	a32.Data[0] = 4194304
	for i := 1; i <= 764; i++ {
		a32.Data[i] = 0.125
	}
	a32.Data[765] = 0.5
	a32.Data[766] = -4194304

	d := MustNewDispatcher(dispatchTestConfig(1))
	c32 := RawMatrix32{Rows: 1, Cols: 1, Stride: 1, Data: make([]float32, 1)}
	if _, err := GemmOn(d, c32, a32, b32, 1, 0); err != nil {
		t.Fatal(err)
	}
	if c32.Data[0] != 0.5 {
		t.Fatalf("standard float32 cancellation = %v, want known native result 0.5", c32.Data[0])
	}

	c32High := RawMatrix32{Rows: 1, Cols: 1, Stride: 1, Data: make([]float32, 1)}
	highDec, err := GemmOn(d, c32High, a32, b32, 1, 0, WithPrecision(PrecisionHigh))
	if err != nil {
		t.Fatal(err)
	}
	if highDec.Precision != PrecisionHigh || highDec.Accumulator == "native-float32" {
		t.Fatalf("high decision not recorded: %+v", highDec)
	}
	if c32High.Data[0] != 96 {
		t.Fatalf("high float32 cancellation = %v, want 96", c32High.Data[0])
	}
	if math.Abs(float64(c32High.Data[0])-96) > 1e-6 {
		t.Fatalf("high float32 error exceeds 1e-6: %v", c32High.Data[0])
	}

	// Float64 analogue: 192 halves are tie-rounded away at 2^52.
	const k64 = 194
	a64 := RawMatrix64{Rows: 1, Cols: k64, Stride: k64, Data: make([]float64, k64)}
	b64 := RawMatrix64{Rows: k64, Cols: 1, Stride: 1, Data: make([]float64, k64)}
	for i := range b64.Data {
		b64.Data[i] = 1
	}
	anchor := float64(1 << 52)
	a64.Data[0] = anchor
	for i := 1; i <= 192; i++ {
		a64.Data[i] = 0.5
	}
	a64.Data[193] = -anchor
	c64High := RawMatrix64{Rows: 1, Cols: 1, Stride: 1, Data: make([]float64, 1)}
	if _, err := GemmOn(d, c64High, a64, b64, 1, 0, WithPrecision(PrecisionHigh)); err != nil {
		t.Fatal(err)
	}
	if c64High.Data[0] != 96 {
		t.Fatalf("high float64 cancellation = %v, want 96", c64High.Data[0])
	}
	if math.Abs(c64High.Data[0]-96) > 1e-12 {
		t.Fatalf("high float64 error exceeds 1e-12: %v", c64High.Data[0])
	}
}

func TestDispatchHighPrecisionLongKParallel(t *testing.T) {
	const k = 2048
	mk := func() (RawMatrix32, RawMatrix32) {
		a := RawMatrix32{Rows: 1, Cols: k, Stride: k, Data: make([]float32, k)}
		b := RawMatrix32{Rows: k, Cols: 1, Stride: 1, Data: make([]float32, k)}
		for i := range b.Data {
			b.Data[i] = 1
		}
		a.Data[0] = 4194304
		// Split the small terms across two deterministic K partitions.
		for i := 0; i < 500; i++ {
			a.Data[1+i] = 0.125
		}
		for i := 0; i < 264; i++ {
			a.Data[1024+i] = 0.125
		}
		a.Data[1300] = 0.5
		a.Data[1301] = -4194304
		return a, b
	}

	var want uint32
	for _, workers := range []int{1, 2, 4} {
		cfg := dispatchTestConfig(workers)
		cfg.MemoryBudget = 1 << 20
		d := MustNewDispatcher(cfg)
		a, b := mk()
		c := RawMatrix32{Rows: 1, Cols: 1, Stride: 1, Data: make([]float32, 1)}
		dec, err := GemmOn(d, c, a, b, 1, 0, WithPrecision(PrecisionHigh))
		if err != nil {
			t.Fatal(err)
		}
		if dec.KPartitions < 2 {
			t.Fatalf("KPartitions=%d want parallel partitions", dec.KPartitions)
		}
		if workers > 1 && dec.Workers < 2 {
			t.Fatalf("workers=%d want K parallelism", dec.Workers)
		}
		if math.Abs(float64(c.Data[0])-96) > 1e-6 {
			t.Fatalf("workers=%d high long-K result=%v want 96", workers, c.Data[0])
		}
		if want == 0 {
			want = math.Float32bits(c.Data[0])
		} else if gotBits := math.Float32bits(c.Data[0]); gotBits != want {
			t.Fatalf("workers=%d bits %08x differ from 1-worker %08x", workers, gotBits, want)
		}
		if workers == 4 && dec.Route != RouteGEMMWorkspace {
			t.Fatalf("route=%s want exact workspace", dec.Route)
		}
	}
}

func TestDispatchAdaptiveChunksFollowShapeAndBudget(t *testing.T) {
	cfg := DefaultDispatchConfig()
	cfg.MaxWorkers = 4
	cfg.MemoryBudget = 1 << 20
	plan := planGemm(cfg, PrecisionStandard, 512, 512, 512, false, false, 8)
	if plan.dec.Chunk.M != defaultChunkM || plan.dec.Chunk.N != defaultChunkN || plan.dec.Chunk.K != defaultChunkK {
		t.Fatalf("large-shape auto chunks=%+v want defaults %d,%d,%d", plan.dec.Chunk, defaultChunkM, defaultChunkN, defaultChunkK)
	}

	small := planGemm(cfg, PrecisionStandard, 3, 5, 7, false, false, 8)
	if small.dec.Chunk.M != 3 || small.dec.Chunk.N != 5 || small.dec.Chunk.K != 7 {
		t.Fatalf("small-shape chunks=%+v want dimension-sized", small.dec.Chunk)
	}

	low := DefaultDispatchConfig()
	low.MaxWorkers = 4
	low.MemoryBudget = 4096
	lowPlan := planGemm(low, PrecisionStandard, 512, 512, 512, false, false, 8)
	if lowPlan.dec.Chunk.K >= defaultChunkK && lowPlan.dec.Chunk.M >= defaultChunkM {
		t.Fatalf("low-budget chunks=%+v want adaptation", lowPlan.dec.Chunk)
	}
	if chunkPanelElements(lowPlan.dec.Chunk.M, lowPlan.dec.Chunk.N, lowPlan.dec.Chunk.K)*8 > low.MemoryBudget &&
		(lowPlan.dec.Chunk.M > 1 || lowPlan.dec.Chunk.N > 1 || lowPlan.dec.Chunk.K > 1) {
		t.Fatalf("low-budget chunk panel %+v does not fit budget", lowPlan.dec.Chunk)
	}

	explicit := DefaultDispatchConfig()
	explicit.ChunkM, explicit.ChunkN, explicit.ChunkK = 13, 17, 19
	explicit.AdaptiveChunks = false
	explicit.MaxWorkers = 2
	explicit.MemoryBudget = 1
	explicitPlan := planGemm(explicit, PrecisionStandard, 512, 512, 512, false, false, 8)
	if explicitPlan.dec.Chunk.M != 13 || explicitPlan.dec.Chunk.N != 17 || explicitPlan.dec.Chunk.K != 19 {
		t.Fatalf("explicit chunks overridden by budget: %+v", explicitPlan.dec.Chunk)
	}
}

func TestDispatchScalarSpecialValues(t *testing.T) {
	a := dispatchDense[float64](3, 2, 0, 81)
	b := dispatchDense[float64](2, 4, 0, 82)
	initial := []float64{1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12}
	for _, alpha := range []float64{0, 1} {
		for _, beta := range []float64{0, 1} {
			cfg := dispatchTestConfig(1)
			d := MustNewDispatcher(cfg)
			c := RawMatrix64{Rows: 3, Cols: 4, Stride: 4, Data: append([]float64(nil), initial...)}
			if _, err := GemmOn(d, c, a, b, alpha, beta); err != nil {
				t.Fatal(err)
			}
			want := dispatchNaive(3, 4, 2, a, b, alpha, beta, append([]float64(nil), initial...))
			dispatchRequireMatrixBits(t, fmt.Sprintf("alpha=%v beta=%v", alpha, beta), c, want)
		}
	}

	// K=0 has empty dot products and retains beta*C.
	emptyA := RawMatrix64{Rows: 3, Cols: 0}
	emptyB := RawMatrix64{Rows: 0, Cols: 4}
	c := RawMatrix64{Rows: 3, Cols: 4, Stride: 4, Data: append([]float64(nil), initial...)}
	d := MustNewDispatcher(dispatchTestConfig(1))
	if _, err := GemmOn(d, c, emptyA, emptyB, 1, 1); err != nil {
		t.Fatal(err)
	}
	dispatchRequireMatrixBits(t, "k-zero-beta-one", c, initial)
}

func TestDispatchOrdinaryPrecisionTolerances(t *testing.T) {
	for _, high := range []bool{false, true} {
		opts := []CallOption{}
		if high {
			opts = append(opts, WithPrecision(PrecisionHigh))
		}
		d := MustNewDispatcher(dispatchTestConfig(4))
		a32 := dispatchDense[float32](31, 37, 0, 41)
		b32 := dispatchDense[float32](37, 29, 0, 42)
		c32 := RawMatrix32{Rows: 31, Cols: 29, Stride: 29, Data: make([]float32, 31*29)}
		if _, err := GemmOn(d, c32, a32, b32, 1, 0, opts...); err != nil {
			t.Fatal(err)
		}
		ref32 := dispatchNaive(31, 29, 37, a32, b32, float32(1), float32(0), nil)
		for i := range c32.Data {
			dispatchApproxEqual(t, "float32 ordinary", float64(c32.Data[i]), float64(ref32[i]), 2e-4)
		}

		a64 := dispatchDense[float64](23, 41, 0, 43)
		b64 := dispatchDense[float64](41, 19, 0, 44)
		c64 := RawMatrix64{Rows: 23, Cols: 19, Stride: 19, Data: make([]float64, 23*19)}
		if _, err := GemmOn(d, c64, a64, b64, 1, 0, opts...); err != nil {
			t.Fatal(err)
		}
		ref64 := dispatchNaive(23, 19, 41, a64, b64, float64(1), float64(0), nil)
		tol := 1e-11
		if high {
			tol = 1e-12
		}
		for i := range c64.Data {
			dispatchApproxEqual(t, "float64 ordinary", c64.Data[i], ref64[i], tol)
		}
	}
}

func TestDispatchBudgetFallbackBitwise(t *testing.T) {
	const k = 5000
	a := dispatchDense[float64](1, k, 0, 51)
	b := dispatchDense[float64](k, 1, 0, 52)

	richCfg := dispatchTestConfig(2)
	richCfg.MemoryBudget = 4096
	rich := MustNewDispatcher(richCfg)
	cRich := RawMatrix64{Rows: 1, Cols: 1, Stride: 1, Data: make([]float64, 1)}
	richDec, err := GemmOn(rich, cRich, a, b, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if richDec.Route != RouteGEMMWorkspace {
		t.Fatalf("rich route=%s want workspace", richDec.Route)
	}

	lowCfg := dispatchTestConfig(2)
	lowCfg.MemoryBudget = 1
	low := MustNewDispatcher(lowCfg)
	cLow := RawMatrix64{Rows: 1, Cols: 1, Stride: 1, Data: make([]float64, 1)}
	lowDec, err := GemmOn(low, cLow, a, b, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if lowDec.Route != RouteGEMMDirect {
		t.Fatalf("low route=%s want direct", lowDec.Route)
	}
	if !lowDec.Degraded || lowDec.FallbackReason != FallbackMemoryBudget {
		t.Fatalf("degradation = %+v want memory-budget", lowDec)
	}
	if lowDec.Workers != 1 || lowDec.Goroutines != 0 {
		t.Fatalf("fallback workers/goroutines=%d/%d want 1/0", lowDec.Workers, lowDec.Goroutines)
	}
	if math.Float64bits(cLow.Data[0]) != math.Float64bits(cRich.Data[0]) {
		t.Fatalf("fallback changed bits: low=%v rich=%v", cLow.Data[0], cRich.Data[0])
	}
	if len(low.pool64.free) != 0 {
		t.Fatal("budget fallback unexpectedly retained workspace")
	}
	if len(rich.pool64.free) != 1 {
		t.Fatalf("workspace pool free length=%d want reused buffer", len(rich.pool64.free))
	}
}

func TestDispatchHighPrecisionBudgetFallbackBitwise(t *testing.T) {
	const k = 5000
	a := dispatchDense[float64](1, k, 0, 53)
	b := dispatchDense[float64](k, 1, 0, 54)
	richCfg := dispatchTestConfig(3)
	richCfg.MemoryBudget = 1 << 20
	rich := MustNewDispatcher(richCfg)
	cRich := RawMatrix64{Rows: 1, Cols: 1, Stride: 1, Data: make([]float64, 1)}
	richDec, err := GemmOn(rich, cRich, a, b, 1, 0, WithPrecision(PrecisionHigh))
	if err != nil {
		t.Fatal(err)
	}
	if richDec.Route != RouteGEMMWorkspace {
		t.Fatalf("rich high route=%s want workspace", richDec.Route)
	}

	lowCfg := dispatchTestConfig(3)
	lowCfg.MemoryBudget = 1
	low := MustNewDispatcher(lowCfg)
	cLow := RawMatrix64{Rows: 1, Cols: 1, Stride: 1, Data: make([]float64, 1)}
	lowDec, err := GemmOn(low, cLow, a, b, 1, 0, WithPrecision(PrecisionHigh))
	if err != nil {
		t.Fatal(err)
	}
	if lowDec.FallbackReason != FallbackMemoryBudget || lowDec.Route != RouteGEMMDirect {
		t.Fatalf("low high decision=%+v", lowDec)
	}
	if math.Float64bits(cLow.Data[0]) != math.Float64bits(cRich.Data[0]) {
		t.Fatal("high precision budget fallback changed bits")
	}
}

func TestDispatchWorkspaceAllocationFailureFallback(t *testing.T) {
	const k = 5000
	a := dispatchDense[float64](1, k, 0, 61)
	b := dispatchDense[float64](k, 1, 0, 62)

	normal := MustNewDispatcher(dispatchTestConfig(2))
	cNormal := RawMatrix64{Rows: 1, Cols: 1, Stride: 1, Data: make([]float64, 1)}
	if _, err := GemmOn(normal, cNormal, a, b, 1, 0); err != nil {
		t.Fatal(err)
	}

	failing := MustNewDispatcher(dispatchTestConfig(2))
	failing.workspaceAllocErr = errors.New("simulated allocation failure")
	cFallback := RawMatrix64{Rows: 1, Cols: 1, Stride: 1, Data: make([]float64, 1)}
	dec, err := GemmOn(failing, cFallback, a, b, 1, 0)
	if err != nil {
		t.Fatalf("allocation failure must degrade, not fail: %v", err)
	}
	if !dec.Degraded || dec.FallbackReason != FallbackWorkspaceAllocation || dec.Route != RouteGEMMDirect {
		t.Fatalf("decision=%+v want allocation fallback", dec)
	}
	if math.Float64bits(cFallback.Data[0]) != math.Float64bits(cNormal.Data[0]) {
		t.Fatal("allocation fallback changed result bits")
	}
	counts := failing.Counts()
	rpc := counts.ByRoutePrecision[RoutePrecisionKey{Route: RouteGEMMDirect, Precision: PrecisionStandard}]
	if counts.Degraded != 1 || rpc.Degraded != 1 {
		t.Fatalf("degraded counts global=%d composite=%+v", counts.Degraded, rpc)
	}
}

func TestDispatchStructuredFailureCount(t *testing.T) {
	d := MustNewDispatcher(dispatchTestConfig(1))
	singular := RawTri64{N: 2, Stride: 2, Upper: true, Data: []float64{0, 1, 0, 0}}
	rhs := RawMatrix64{Rows: 2, Cols: 1, Stride: 1, Data: []float64{1, 2}}
	x := RawMatrix64{Rows: 2, Cols: 1, Stride: 1, Data: make([]float64, 2)}
	dec, err := TriSolveOn(d, x, rhs, singular, false)
	if !errors.Is(err, ErrSingular) {
		t.Fatalf("error=%v want ErrSingular", err)
	}
	if dec.OK || dec.Error == "" {
		t.Fatalf("decision did not record failure: %+v", dec)
	}
	counts := d.Counts()
	if counts.Calls != 1 || counts.Failures != 1 {
		t.Fatalf("counts=%+v want one call and failure", counts)
	}
	rpc := counts.ByRoutePrecision[RoutePrecisionKey{Route: RouteTriSolve, Precision: PrecisionStandard}]
	if rpc.Calls != 1 || rpc.Failures != 1 {
		t.Fatalf("composite counts=%+v", rpc)
	}
	page := d.HistoryPage(1)
	if len(page.Records) != 1 || page.Records[0].OK {
		t.Fatalf("failed decision not observable: %+v", page.Records)
	}
}

func TestDispatchHistoryRingAndInvalidArgs(t *testing.T) {
	d := MustNewDispatcher(func() DispatchConfig {
		c := dispatchTestConfig(2)
		c.HistorySize = 3
		return c
	}())
	if page := d.HistoryPage(0); len(page.Records) != 0 || page.Total != 0 {
		t.Fatalf("fresh history = %+v", page)
	}

	a := dispatchDense[float64](2, 2, 0, 1)
	badB := dispatchDense[float64](3, 2, 0, 2)
	c := RawMatrix64{Rows: 2, Cols: 2, Stride: 2, Data: make([]float64, 4)}
	_, err := GemmOn(d, c, a, badB, 1, 0)
	if err == nil {
		t.Fatal("expected shape error")
	}
	if page := d.HistoryPage(0); page.Total != 0 || len(page.Records) != 0 {
		t.Fatal("invalid arguments left history or count")
	}
	if counts := d.Counts(); counts.Calls != 0 {
		t.Fatalf("invalid call counted: %+v", counts)
	}

	for i := 0; i < 5; i++ {
		b := dispatchDense[float64](2, 2, 0, 100+i)
		if _, err := GemmOn(d, c, a, b, 1, 0); err != nil {
			t.Fatal(err)
		}
	}
	page := d.HistoryPage(0)
	if page.Total != 5 || page.Capacity != 3 || len(page.Records) != 3 {
		t.Fatalf("page=%+v", page)
	}
	if page.Records[0].ID != 3 || page.Records[2].ID != 5 {
		t.Fatalf("ring IDs = %d,%d,%d want 3,4,5", page.Records[0].ID, page.Records[1].ID, page.Records[2].ID)
	}
	limited := d.History(2)
	if len(limited) != 2 || limited[0].ID != 4 {
		t.Fatalf("limit returned %+v", limited)
	}

	d.ResetObservability()
	if page := d.HistoryPage(0); page.Total != 0 || len(page.Records) != 0 {
		t.Fatal("reset did not clear history")
	}
}

func TestDispatchCollectionSwitchAndCountsConcurrent(t *testing.T) {
	cfg := dispatchTestConfig(4)
	cfg.HistorySize = 37
	d := MustNewDispatcher(cfg)

	const goroutines = 12
	const iterations = 20
	sharedA := dispatchDense[float64](17, 19, 0, 1001)
	sharedB := dispatchDense[float64](19, 23, 0, 2002)
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				c := RawMatrix64{Rows: 17, Cols: 23, Stride: 23, Data: make([]float64, 17*23)}
				prec := PrecisionStandard
				opts := []CallOption{}
				if (g+i)%2 == 0 {
					prec = PrecisionHigh
					opts = append(opts, WithPrecision(prec))
				}
				want := make([]float64, 17*23)
				for row := 0; row < 17; row++ {
					for col := 0; col < 23; col++ {
						want[row*23+col] = reduceDot(sharedA, sharedB, row, col, 19, 1, 11, prec)
					}
				}
				dec, err := GemmOn(d, c, sharedA, sharedB, 1, 0, opts...)
				if err != nil {
					t.Errorf("dispatch: %v", err)
					return
				}
				if dec.Workers > 4 {
					t.Errorf("workers %d exceed cap", dec.Workers)
				}
				for idx, w := range want {
					if !dispatchBitsEqual(c.Data[idx], w) {
						t.Errorf("concurrent call g=%d i=%d element %d changed bits", g, i, idx)
						return
					}
				}
			}
		}(g)
	}
	stopQueries := make(chan struct{})
	var queryWG sync.WaitGroup
	queryWG.Add(1)
	go func() {
		defer queryWG.Done()
		for {
			select {
			case <-stopQueries:
				return
			default:
				_ = d.HistoryPage(7)
				_ = d.Counts()
			}
		}
	}()
	wg.Wait()
	close(stopQueries)
	queryWG.Wait()

	total := int64(goroutines * iterations)
	counts := d.Counts()
	if counts.Calls != total {
		t.Fatalf("calls=%d want %d", counts.Calls, total)
	}
	var routeSum, precSum, rpcSum int64
	for _, v := range counts.ByRoute {
		routeSum += v
	}
	for _, v := range counts.ByPrecision {
		precSum += v
	}
	for _, v := range counts.ByRoutePrecision {
		rpcSum += v.Calls
	}
	if routeSum != total || precSum != total || rpcSum != total {
		t.Fatalf("counter sums route=%d precision=%d composite=%d total=%d", routeSum, precSum, rpcSum, total)
	}
	page := d.HistoryPage(0)
	if page.Total != uint64(total) || len(page.Records) > cfg.HistorySize {
		t.Fatalf("history total=%d len=%d", page.Total, len(page.Records))
	}

	offCfg := dispatchTestConfig(2)
	offCfg.Collect = false
	off := MustNewDispatcher(offCfg)
	a := dispatchDense[float64](7, 5, 0, 1)
	b := dispatchDense[float64](5, 3, 0, 2)
	c1 := RawMatrix64{Rows: 7, Cols: 3, Stride: 3, Data: make([]float64, 21)}
	c2 := RawMatrix64{Rows: 7, Cols: 3, Stride: 3, Data: make([]float64, 21)}
	if _, err := GemmOn(off, c1, a, b, 1, 0); err != nil {
		t.Fatal(err)
	}
	on := MustNewDispatcher(dispatchTestConfig(2))
	if _, err := GemmOn(on, c2, a, b, 1, 0); err != nil {
		t.Fatal(err)
	}
	dispatchRequireMatrixBits(t, "collection-off", c1, c2.Data)
	if page := off.HistoryPage(0); len(page.Records) != 0 || page.Total != 1 {
		t.Fatalf("collect=false page=%+v want no records but counters", page)
	}
}

func TestDispatchDenseAdapter(t *testing.T) {
	d := MustNewDispatcher(dispatchTestConfig(2))
	a := NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	b := NewDense(3, 2, []float64{7, 8, 9, 10, 11, 12})
	dst := NewDense(2, 2, nil)
	dec, err := d.DenseMulDispatch(dst, a, b)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Op != "gemm" || dec.M != 2 || dec.N != 2 || dec.K != 3 {
		t.Fatalf("bad decision %+v", dec)
	}
	var want Dense
	want.Mul(a, b)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			dispatchApproxEqual(t, "dense adapter", dst.At(i, j), want.At(i, j), 1e-14)
		}
	}

	// Implicit transpose wrappers are unwrapped and flagged in the decision.
	at := NewDense(3, 2, []float64{1, 4, 2, 5, 3, 6}) // transpose of a
	bt := NewDense(2, 3, []float64{7, 9, 11, 8, 10, 12})
	dstT := NewDense(2, 2, nil)
	decT, err := d.DenseMulDispatch(dstT, at.T(), bt.T())
	if err != nil {
		t.Fatal(err)
	}
	if !decT.TransA || !decT.TransB {
		t.Fatalf("transpose flags not recorded: %+v", decT)
	}
	var wantT Dense
	wantT.Mul(at.T(), bt.T())
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			dispatchApproxEqual(t, "dense transpose adapter", dstT.At(i, j), wantT.At(i, j), 1e-14)
		}
	}
}

func TestDispatchStructuredEntriesRecorded(t *testing.T) {
	d := MustNewDispatcher(dispatchTestConfig(2))

	// Symmetric multiplication.
	symData := []float64{
		2, 1, 0, 0,
		1, 2, 1, 0,
		0, 1, 2, 1,
		0, 0, 1, 2,
	}
	sym := NewSymDense(4, symData)
	b := NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	cSym := RawMatrix64{Rows: 4, Cols: 2, Stride: 2, Data: make([]float64, 8)}
	symDec, err := SymmetricMulOn(d, cSym, RawSymFromSymDense(sym), RawFromDense(b), LeftSide, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if symDec.Route != RouteSymmetric || !symDec.Specialized {
		t.Fatalf("sym decision %+v", symDec)
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 2; j++ {
			var want float64
			for k := 0; k < 4; k++ {
				want += sym.At(i, k) * b.At(k, j)
			}
			dispatchApproxEqual(t, "symmetric", cSym.at(i, j), want, 1e-12)
		}
	}

	// Triangular multiplication.
	triA := NewTriDense(3, Upper, []float64{
		1, 2, 3,
		0, 4, 5,
		0, 0, 6,
	})
	triB := NewTriDense(3, Upper, []float64{
		2, 0, 1,
		0, 1, 2,
		0, 0, 3,
	})
	triDst := RawTri64{N: 3, Stride: 3, Upper: true, Data: make([]float64, 9)}
	triDec, err := TriangularMulOn(d, triDst, RawTriFromTriDense(triA), RawTriFromTriDense(triB))
	if err != nil {
		t.Fatal(err)
	}
	if triDec.Route != RouteTriangularMul || !triDec.Specialized {
		t.Fatalf("tri decision %+v", triDec)
	}
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			var want float64
			for k := i; k <= j; k++ {
				want += triA.At(i, k) * triB.At(k, j)
			}
			dispatchApproxEqual(t, "trimul", triDst.at(i, j), want, 1e-12)
		}
	}

	// Rank-k update.
	rkC := RawSym64{N: 4, Stride: 4, Upper: true, Data: make([]float64, 16)}
	rkA := dispatchDense[float64](4, 2, 0, 71)
	rkDec, err := RankKOn(d, rkC, rkA, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if rkDec.Route != RouteRankK || !rkDec.Specialized {
		t.Fatalf("rank-k decision %+v", rkDec)
	}
	for i := 0; i < 4; i++ {
		for j := i; j < 4; j++ {
			var want float64
			for k := 0; k < 2; k++ {
				want += rkA.at(i, k) * rkA.at(j, k)
			}
			dispatchApproxEqual(t, "rank-k", rkC.at(i, j), want, 1e-12)
		}
	}

	// Triangular solve.
	triSolveT := RawTri64{N: 3, Stride: 3, Upper: true, Data: []float64{
		2, 1, 0,
		0, 3, 1,
		0, 0, 4,
	}}
	rhs := RawMatrix64{Rows: 3, Cols: 2, Stride: 2, Data: []float64{1, 2, 3, 4, 5, 6}}
	x := RawMatrix64{Rows: 3, Cols: 2, Stride: 2, Data: make([]float64, 6)}
	solveDec, err := TriSolveOn(d, x, rhs, triSolveT, false, WithPrecision(PrecisionHigh))
	if err != nil {
		t.Fatal(err)
	}
	if solveDec.Route != RouteTriSolve || !solveDec.Specialized {
		t.Fatalf("solve decision %+v", solveDec)
	}
	for col := 0; col < 2; col++ {
		for i := 0; i < 3; i++ {
			var lhs float64
			for k := i; k < 3; k++ {
				lhs += triSolveT.at(i, k) * x.at(k, col)
			}
			dispatchApproxEqual(t, "trsm residual", lhs, rhs.at(i, col), 1e-10)
		}
	}

	counts := d.Counts()
	for _, route := range []Route{RouteSymmetric, RouteTriangularMul, RouteRankK, RouteTriSolve} {
		if counts.ByRoute[route] != 1 {
			t.Fatalf("route %s count=%d want 1; counts=%+v", route, counts.ByRoute[route], counts.ByRoute)
		}
	}
	page := d.HistoryPage(10)
	if len(page.Records) != 4 {
		t.Fatalf("structured records=%d want 4", len(page.Records))
	}
}
