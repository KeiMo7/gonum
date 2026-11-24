// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"sync"

	"gonum.org/v1/gonum/blas"
)

// Side selects whether a structured matrix is on the left or right of a product.
type Side uint8

const (
	// LeftSide computes A*B.
	LeftSide Side = iota + 1
	// RightSide computes B*A.
	RightSide
)

// RawSymN is a stride-aware symmetric descriptor using BLAS triangle storage.
type RawSymN[T Numeric] struct {
	N      int
	Stride int
	Data   []T
	Upper  bool
}

// RawSym64 is the float64 specialization.
type RawSym64 = RawSymN[float64]

// RawSym32 is the float32 specialization.
type RawSym32 = RawSymN[float32]

// RawSymFromSymDense aliases s in a generic raw descriptor.
func RawSymFromSymDense(s *SymDense) RawSym64 {
	if s == nil {
		return RawSym64{}
	}
	return RawSym64{N: s.mat.N, Stride: s.mat.Stride, Data: s.mat.Data, Upper: s.mat.Uplo == blas.Upper}
}

func (s RawSymN[T]) normalized(name string) (RawSymN[T], error) {
	if s.N < 0 {
		return s, fmt.Errorf("%w: %s has negative size %d", ErrShape, name, s.N)
	}
	if s.Stride == 0 && s.N > 0 {
		s.Stride = s.N
	}
	if s.N > 0 && s.Stride < s.N {
		return s, fmt.Errorf("%w: %s stride %d is less than N %d", ErrShape, name, s.Stride, s.N)
	}
	if s.N > 0 {
		need := (s.N-1)*s.Stride + s.N
		if need > len(s.Data) {
			return s, fmt.Errorf("%w: %s storage length %d is too short; need %d", ErrShape, name, len(s.Data), need)
		}
	}
	return s, nil
}

func (s RawSymN[T]) at(i, j int) T {
	if s.Upper {
		if i > j {
			i, j = j, i
		}
		return s.Data[i*s.Stride+j]
	}
	if i < j {
		i, j = j, i
	}
	return s.Data[i*s.Stride+j]
}

func (s RawSymN[T]) set(i, j int, v T) {
	if s.Upper {
		if i > j {
			i, j = j, i
		}
		s.Data[i*s.Stride+j] = v
		return
	}
	if i < j {
		i, j = j, i
	}
	s.Data[i*s.Stride+j] = v
}

// RawTriN is a stride-aware triangular descriptor.
type RawTriN[T Numeric] struct {
	N      int
	Stride int
	Data   []T
	Upper  bool
	Unit   bool
}

// RawTri64 is the float64 specialization.
type RawTri64 = RawTriN[float64]

// RawTri32 is the float32 specialization.
type RawTri32 = RawTriN[float32]

// RawTriFromTriDense aliases t in a generic raw descriptor.
func RawTriFromTriDense(t *TriDense) RawTri64 {
	if t == nil {
		return RawTri64{}
	}
	_, kind := t.Triangle()
	return RawTri64{
		N:      t.mat.N,
		Stride: t.mat.Stride,
		Data:   t.mat.Data,
		Upper:  kind == Upper,
		Unit:   t.mat.Diag == blas.Unit,
	}
}

func (t RawTriN[T]) normalized(name string) (RawTriN[T], error) {
	if t.N < 0 {
		return t, fmt.Errorf("%w: %s has negative size %d", ErrShape, name, t.N)
	}
	if t.Stride == 0 && t.N > 0 {
		t.Stride = t.N
	}
	if t.N > 0 && t.Stride < t.N {
		return t, fmt.Errorf("%w: %s stride %d is less than N %d", ErrShape, name, t.Stride, t.N)
	}
	if t.N > 0 {
		need := (t.N-1)*t.Stride + t.N
		if need > len(t.Data) {
			return t, fmt.Errorf("%w: %s storage length %d is too short; need %d", ErrShape, name, len(t.Data), need)
		}
	}
	return t, nil
}

func (t RawTriN[T]) at(i, j int) T {
	if t.Unit && i == j {
		var one T = 1
		return one
	}
	return t.Data[i*t.Stride+j]
}

func (t RawTriN[T]) set(i, j int, v T) {
	if t.Unit && i == j {
		return
	}
	t.Data[i*t.Stride+j] = v
}

func planStructured(cfg DispatchConfig, p Precision, op string, route Route, m, n, k, elemSize int) Decision {
	base := planGemm(cfg, p, m, n, k, false, false, elemSize)
	d := base.dec
	d.Op = op
	d.Route = route
	d.Specialized = true
	d.KPartitions = 1
	// Structured reductions use one ascending sum per output/right-hand side.
	if d.Workers > 1 && base.tiles < 2 && m*n >= 2 {
		d.Workers = minInt(cfg.MaxWorkers, m*n)
		d.Goroutines = d.Workers - 1
	} else if d.Workers > 1 && base.tiles < 2 {
		d.Workers = 1
		d.Goroutines = 0
	}
	d.Accumulator = accumulatorName(p)
	if p == PrecisionStandard {
		d.Accumulator = "native-element"
	} else {
		d.Accumulator = "exact-superaccumulator"
	}
	return d
}

func runIndexedJobs(workers, jobs int, fn func(job int)) {
	if jobs <= 0 {
		return
	}
	if workers <= 1 || jobs == 1 {
		for j := 0; j < jobs; j++ {
			fn(j)
		}
		return
	}
	next := make(chan int, jobs)
	for j := 0; j < jobs; j++ {
		next <- j
	}
	close(next)
	var wg sync.WaitGroup
	work := func() {
		for j := range next {
			fn(j)
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
}

func commitDecision(d *Dispatcher, dec Decision, err error) (Decision, error) {
	dec = d.commit(dec, err)
	return dec, err
}

// SymmetricMulN runs SymmetricMulOn on the default dispatcher.
func SymmetricMulN[T Numeric](c RawMatrixN[T], a RawSymN[T], b RawMatrixN[T], side Side, alpha, beta T, opts ...CallOption) (Decision, error) {
	return SymmetricMulOn(defaultDispatcher, c, a, b, side, alpha, beta, opts...)
}

// TriangularMulN runs TriangularMulOn on the default dispatcher.
func TriangularMulN[T Numeric](dst RawTriN[T], a, b RawTriN[T], opts ...CallOption) (Decision, error) {
	return TriangularMulOn(defaultDispatcher, dst, a, b, opts...)
}

// RankKN runs RankKOn on the default dispatcher.
func RankKN[T Numeric](c RawSymN[T], a RawMatrixN[T], alpha, beta T, opts ...CallOption) (Decision, error) {
	return RankKOn(defaultDispatcher, c, a, alpha, beta, opts...)
}

// TriSolveN runs TriSolveOn on the default dispatcher.
func TriSolveN[T Numeric](dst, b RawMatrixN[T], t RawTriN[T], transT bool, opts ...CallOption) (Decision, error) {
	return TriSolveOn(defaultDispatcher, dst, b, t, transT, opts...)
}

// SymmetricMulOn computes a structure-aware symmetric product on d.
func SymmetricMulOn[T Numeric](d *Dispatcher, c RawMatrixN[T], a RawSymN[T], b RawMatrixN[T], side Side, alpha, beta T, opts ...CallOption) (Decision, error) {
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
	bRows, bCols := b.logicalDims()
	var m, n int
	switch side {
	case LeftSide:
		if bRows != a.N {
			return Decision{}, fmt.Errorf("%w: symmetric A is %d×%d but B has %d rows", ErrShape, a.N, a.N, bRows)
		}
		m, n = a.N, bCols
	case RightSide:
		if bCols != a.N {
			return Decision{}, fmt.Errorf("%w: B columns %d do not match symmetric A size %d", ErrShape, bCols, a.N)
		}
		m, n = bRows, a.N
	default:
		return Decision{}, fmt.Errorf("%w: unknown multiplication side %d", ErrShape, side)
	}
	if cm, cn := c.logicalDims(); cm != m || cn != n {
		return Decision{}, fmt.Errorf("%w: C is %d×%d but product is %d×%d", ErrShape, cm, cn, m, n)
	}

	dec := planStructured(cfg, p, "symmetric-mul", RouteSymmetric, m, n, a.N, numericBytes[T]())
	err = symmetricMulRun(dec, c, a, b, side, alpha, beta)
	return commitDecision(d, dec, err)
}

func symmetricMulRun[T Numeric](dec Decision, c RawMatrixN[T], a RawSymN[T], b RawMatrixN[T], side Side, alpha, beta T) error {
	m, n, k := dec.M, dec.N, dec.K
	jobs := m * n
	runIndexedJobs(dec.Workers, jobs, func(job int) {
		i := job / n
		j := job % n
		var sum T
		var exact exactAccumulator
		useExact := dec.Precision == PrecisionHigh
		if useExact {
			exact.reset()
		}
		for q := 0; q < k; q++ {
			var x, y T
			if side == LeftSide {
				x = a.at(i, q)
				y = b.at(q, j)
			} else {
				x = b.at(i, q)
				y = a.at(q, j)
			}
			if useExact {
				if !exactAddProduct(&exact, x, y) {
					useExact = false
				}
			}
			if !useExact {
				sum += x * y
			}
		}
		if dec.Precision == PrecisionHigh && useExact {
			sum = exactValueT[T](&exact)
		}
		writeGemmCell(c, i, j, alpha, beta, sum)
	})
	return nil
}

// TriangularMulOn computes the product of two triangles with the same kind.
func TriangularMulOn[T Numeric](d *Dispatcher, dst RawTriN[T], a, b RawTriN[T], opts ...CallOption) (Decision, error) {
	if d == nil {
		d = defaultDispatcher
	}
	cfg := d.configSnapshot()
	p, err := resolvePrecision(cfg, opts)
	if err != nil {
		return Decision{}, err
	}
	dst, err = dst.normalized("dst")
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
	if a.N != b.N || a.N != dst.N {
		err := fmt.Errorf("%w: triangular sizes dst=%d a=%d b=%d", ErrShape, dst.N, a.N, b.N)
		return Decision{}, err
	}
	if a.Upper != b.Upper || a.Upper != dst.Upper {
		return Decision{}, ErrTriangle
	}
	n := a.N
	dec := planStructured(cfg, p, "triangular-mul", RouteTriangularMul, n, n, n, numericBytes[T]())
	jobs := n * (n + 1) / 2
	runIndexedJobs(dec.Workers, jobs, func(job int) {
		// Map linear triangle index to row/column. n is small enough for the
		// direct sqrt-free search used here; jobs are still independent.
		var i, j int
		if a.Upper {
			acc := 0
			for i = 0; i < n; i++ {
				rowJobs := n - i
				if job < acc+rowJobs {
					j = i + job - acc
					break
				}
				acc += rowJobs
			}
		} else {
			acc := 0
			for i = 0; i < n; i++ {
				rowJobs := i + 1
				if job < acc+rowJobs {
					j = job - acc
					break
				}
				acc += rowJobs
			}
		}

		var sum T
		var exact exactAccumulator
		useExact := p == PrecisionHigh
		if useExact {
			exact.reset()
		}
		k0, k1 := 0, n-1
		if a.Upper {
			k0, k1 = i, j
		} else {
			k0, k1 = j, i
		}
		for k := k0; k <= k1; k++ {
			x := a.at(i, k)
			y := b.at(k, j)
			if useExact && !exactAddProduct(&exact, x, y) {
				useExact = false
			}
			if !useExact {
				sum += x * y
			}
		}
		if p == PrecisionHigh && useExact {
			sum = exactValueT[T](&exact)
		}
		dst.set(i, j, sum)
	})
	return commitDecision(d, dec, nil)
}

// RankKOn performs C = alpha*A*Aᵀ + beta*C on a symmetric descriptor.
func RankKOn[T Numeric](d *Dispatcher, c RawSymN[T], a RawMatrixN[T], alpha, beta T, opts ...CallOption) (Decision, error) {
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
	ar, ak := a.logicalDims()
	if ar != c.N {
		return Decision{}, fmt.Errorf("%w: A has %d rows but symmetric C size %d", ErrShape, ar, c.N)
	}
	n, k := c.N, ak
	dec := planStructured(cfg, p, "rank-k", RouteRankK, n, n, k, numericBytes[T]())
	jobs := n * (n + 1) / 2
	runIndexedJobs(dec.Workers, jobs, func(job int) {
		var i, j int
		if c.Upper {
			acc := 0
			for i = 0; i < n; i++ {
				rowJobs := n - i
				if job < acc+rowJobs {
					j = i + job - acc
					break
				}
				acc += rowJobs
			}
		} else {
			acc := 0
			for i = 0; i < n; i++ {
				rowJobs := i + 1
				if job < acc+rowJobs {
					j = job - acc
					break
				}
				acc += rowJobs
			}
		}
		var dot T
		var exact exactAccumulator
		useExact := p == PrecisionHigh
		if useExact {
			exact.reset()
		}
		for q := 0; q < k; q++ {
			x := a.at(i, q)
			y := a.at(j, q)
			if useExact && !exactAddProduct(&exact, x, y) {
				useExact = false
			}
			if !useExact {
				dot += x * y
			}
		}
		if p == PrecisionHigh && useExact {
			dot = exactValueT[T](&exact)
		}
		old := c.at(i, j)
		var v T
		if isOneT(alpha) {
			v = dot
		} else {
			v = alpha * dot
		}
		if isZeroT(beta) {
			c.set(i, j, v)
		} else if isOneT(beta) {
			c.set(i, j, v+old)
		} else {
			c.set(i, j, v+beta*old)
		}
	})
	return commitDecision(d, dec, nil)
}

// TriSolveOn solves op(T)*X = B and writes X to dst. dst may alias B only when
// their storage is identical and no separate output is required by the caller.
func TriSolveOn[T Numeric](d *Dispatcher, dst, b RawMatrixN[T], t RawTriN[T], transT bool, opts ...CallOption) (Decision, error) {
	if d == nil {
		d = defaultDispatcher
	}
	cfg := d.configSnapshot()
	p, err := resolvePrecision(cfg, opts)
	if err != nil {
		return Decision{}, err
	}
	dst, err = dst.normalized("dst")
	if err != nil {
		return Decision{}, err
	}
	b, err = b.normalized("b")
	if err != nil {
		return Decision{}, err
	}
	t, err = t.normalized("t")
	if err != nil {
		return Decision{}, err
	}
	br, bc := b.logicalDims()
	dr, dc := dst.logicalDims()
	if br != t.N {
		return Decision{}, fmt.Errorf("%w: B has %d rows but T size %d", ErrShape, br, t.N)
	}
	if dr != t.N || dc != bc {
		return Decision{}, fmt.Errorf("%w: X is %d×%d but must be %d×%d", ErrShape, dr, dc, t.N, bc)
	}

	dec := planStructured(cfg, p, "triangular-solve", RouteTriSolve, t.N, bc, t.N, numericBytes[T]())
	singular := false
	var singularMu sync.Mutex
	runIndexedJobs(dec.Workers, bc, func(col int) {
		// Copy B column into dst first so dst may be a distinct descriptor.
		for i := 0; i < t.N; i++ {
			dst.set(i, col, b.at(i, col))
		}
		for step := 0; step < t.N; step++ {
			var i int
			var indices []int
			// Determine substitution direction for op(T).
			forward := false
			if !transT {
				forward = !t.Upper
			} else {
				forward = t.Upper // Tᵀ of upper is lower.
			}
			if forward {
				i = step
				if !transT {
					indices = indices[:0]
					for k := 0; k < i; k++ {
						indices = append(indices, k)
					}
				} else {
					// (Tᵀ X): contributions T[k,i] X[k], k<i for upper T.
					indices = indices[:0]
					for k := 0; k < i; k++ {
						indices = append(indices, k)
					}
				}
			} else {
				i = t.N - 1 - step
				indices = indices[:0]
				if !transT {
					for k := i + 1; k < t.N; k++ {
						indices = append(indices, k)
					}
				} else {
					// T lower transposed: T[k,i], k>i.
					for k := i + 1; k < t.N; k++ {
						indices = append(indices, k)
					}
				}
			}

			var sum T
			var exact exactAccumulator
			useExact := p == PrecisionHigh
			if useExact {
				exact.reset()
			}
			for _, k := range indices {
				var coeff T
				if !transT {
					coeff = t.at(i, k)
				} else {
					coeff = t.at(k, i)
				}
				x := dst.at(k, col)
				if useExact && !exactAddProduct(&exact, coeff, x) {
					useExact = false
				}
				if !useExact {
					sum += coeff * x
				}
			}
			if p == PrecisionHigh && useExact {
				sum = exactValueT[T](&exact)
			}
			diag := t.at(i, i)
			if isZeroT(diag) {
				singularMu.Lock()
				singular = true
				singularMu.Unlock()
				return
			}
			rhs := b.at(i, col)
			dst.set(i, col, (rhs-sum)/diag)
		}
	})
	var runErr error
	if singular {
		runErr = ErrSingular
	}
	return commitDecision(d, dec, runErr)
}

// DenseMulDispatch runs the unified generic GEMM route for two *Dense values.
// Transpose wrappers are unwrapped using mat's existing semantics.
func (d *Dispatcher) DenseMulDispatch(dst *Dense, a, b Matrix, opts ...CallOption) (Decision, error) {
	if d == nil {
		d = defaultDispatcher
	}
	if dst == nil {
		return Decision{}, fmt.Errorf("%w: nil destination", ErrShape)
	}
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ac != br {
		return Decision{}, ErrShape
	}
	dst.reuseAsNonZeroed(ar, bc)

	aU, aTrans := untransposeExtract(a)
	bU, bTrans := untransposeExtract(b)
	ad, aok := aU.(*Dense)
	bd, bok := bU.(*Dense)
	if !aok || !bok {
		return Decision{}, fmt.Errorf("mat: dispatch DenseMulDispatch requires dense raw operands")
	}
	ra := RawFromDense(ad)
	ra.Trans = aTrans
	rb := RawFromDense(bd)
	rb.Trans = bTrans
	rc := RawFromDense(dst)
	one := float64(1)
	zero := float64(0)
	return GemmOn(d, rc, ra, rb, one, zero, opts...)
}

// DispatchDenseMul runs DenseMulDispatch on the default dispatcher.
func DispatchDenseMul(dst *Dense, a, b Matrix, opts ...CallOption) (Decision, error) {
	return defaultDispatcher.DenseMulDispatch(dst, a, b, opts...)
}
