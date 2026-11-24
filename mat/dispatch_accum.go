// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"
	"math/big"
)

// floatParts decomposes a finite IEEE scalar as mantissa*2^exp. The returned
// mantissa is signed and fits in int64 for both float32 and float64.
func floatParts[T Numeric](x T) (mant int64, exp int, finite bool) {
	switch v := any(x).(type) {
	case float32:
		bits := math.Float32bits(v)
		sign := bits >> 31
		expBits := (bits >> 23) & 0xff
		frac := bits & (1<<23 - 1)
		if expBits == 0xff {
			return 0, 0, false
		}
		var m int64
		var e int
		if expBits == 0 {
			if frac == 0 {
				return 0, 0, true
			}
			m = int64(frac)
			e = -149 // 1-bias-fracBits = 1-127-23
		} else {
			m = int64((1 << 23) | frac)
			e = int(expBits) - 127 - 23
		}
		if sign != 0 {
			m = -m
		}
		return m, e, true
	case float64:
		bits := math.Float64bits(v)
		sign := bits >> 63
		expBits := (bits >> 52) & 0x7ff
		frac := bits & (1<<52 - 1)
		if expBits == 0x7ff {
			return 0, 0, false
		}
		var m int64
		var e int
		if expBits == 0 {
			if frac == 0 {
				return 0, 0, true
			}
			m = int64(frac)
			e = -1074
		} else {
			m = int64((1 << 52) | frac)
			e = int(expBits) - 1023 - 52
		}
		if sign != 0 {
			m = -m
		}
		return m, e, true
	default:
		return 0, 0, false
	}
}

type exactAccumulator struct {
	sum    big.Int
	term   big.Int
	left   big.Int
	shift  int
	empty  bool
	finite bool
}

const exactAccumulatorEstimateBytes = 1024

func (e *exactAccumulator) reset() {
	e.sum.SetInt64(0)
	e.term.SetInt64(0)
	e.left.SetInt64(0)
	e.shift = 0
	e.empty = true
	e.finite = true
}

func mergeExactAccumulator(dst *exactAccumulator, src *exactAccumulator) {
	if src == nil || src.empty {
		return
	}
	dst.addBig(&src.sum, src.shift)
}

// exactAddProduct adds xa*xb exactly. It returns false for non-finite inputs
// so the caller can preserve IEEE special-value behavior with a hardware path.
func exactAddProduct[T Numeric](e *exactAccumulator, xa, xb T) bool {
	ma, ea, ok := floatParts(xa)
	if !ok {
		return false
	}
	mb, eb, ok := floatParts(xb)
	if !ok {
		return false
	}
	if ma == 0 || mb == 0 {
		return true
	}

	var a, b, prod big.Int
	a.SetInt64(ma)
	b.SetInt64(mb)
	prod.Mul(&a, &b)
	exp := ea + eb
	e.addBig(&prod, exp)
	return true
}

func exactAddFloat[T Numeric](e *exactAccumulator, x T) bool {
	m, exp, ok := floatParts(x)
	if !ok {
		return false
	}
	if m == 0 {
		return true
	}
	var v big.Int
	v.SetInt64(m)
	e.addBig(&v, exp)
	return true
}

func (e *exactAccumulator) addBig(v *big.Int, exp int) {
	if e.empty {
		e.sum.Set(v)
		e.shift = exp
		e.empty = false
		return
	}
	switch {
	case exp == e.shift:
		e.sum.Add(&e.sum, v)
	case exp > e.shift:
		e.term.Lsh(v, uint(exp-e.shift))
		e.sum.Add(&e.sum, &e.term)
	default:
		// Move the existing sum to the new, finer scale.
		e.left.Lsh(&e.sum, uint(e.shift-exp))
		e.sum.Set(&e.left)
		e.shift = exp
		e.sum.Add(&e.sum, v)
	}
}

func (e *exactAccumulator) valueFloat32() float32 {
	if e.empty || e.sum.Sign() == 0 {
		return 0
	}
	mant := new(big.Float).SetPrec(uint(e.sum.BitLen() + 1)).SetInt(&e.sum)
	out := new(big.Float).SetPrec(24)
	out.SetMantExp(mant, e.shift)
	v, _ := out.Float32()
	return v
}

func (e *exactAccumulator) valueFloat64() float64 {
	if e.empty || e.sum.Sign() == 0 {
		return 0
	}
	mant := new(big.Float).SetPrec(uint(e.sum.BitLen() + 1)).SetInt(&e.sum)
	out := new(big.Float).SetPrec(53)
	out.SetMantExp(mant, e.shift)
	v, _ := out.Float64()
	return v
}

func exactValueT[T Numeric](e *exactAccumulator) T {
	var zero T
	switch any(zero).(type) {
	case float32:
		return any(e.valueFloat32()).(T)
	case float64:
		return any(e.valueFloat64()).(T)
	default:
		return zero
	}
}

func exactSegmentInto[T Numeric](acc *exactAccumulator, a, b RawMatrixN[T], row, col, start, end int) bool {
	for k := start; k < end; k++ {
		if !exactAddProduct(acc, a.at(row, k), b.at(k, col)) {
			acc.finite = false
			return false
		}
	}
	return true
}

func exactSegment[T Numeric](a, b RawMatrixN[T], row, col, start, end int) (T, bool) {
	var acc exactAccumulator
	acc.reset()
	if !exactSegmentInto(&acc, a, b, row, col, start, end) {
		return 0, false
	}
	return exactValueT[T](&acc), true
}

func combinePartials[T Numeric](partials []T, p Precision) T {
	if p != PrecisionHigh {
		var total T
		for _, v := range partials {
			total += v
		}
		return total
	}
	var acc exactAccumulator
	acc.reset()
	for _, v := range partials {
		if !exactAddFloat(&acc, v) {
			// Segment special values should already have followed the hardware
			// path; this fallback keeps special-value propagation safe.
			var total T
			for _, x := range partials {
				total += x
			}
			return total
		}
	}
	return exactValueT[T](&acc)
}
