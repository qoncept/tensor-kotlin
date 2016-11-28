package jp.co.qoncept.tensorkotlin

import java.util.*

class Tensor(val shape: Shape, val elements: FloatArray) {
    constructor(shape: Shape, element: Float = 0.0f) : this(shape, floatArrayOf(shape.volume, element)) {
    }

    internal fun index(indices: IntArray): Int {
        assert({ indices.size == shape.dimensions.size }, { "`indices.size` must be ${shape.dimensions.size}: ${indices.size}" })
        return shape.dimensions.zip(indices).fold(0) { a, x ->
            assert({ 0 <= x.second && x.second < x.first }, { "Illegal index: indices = ${indices}, shape = ${shape}" })
            a * x.first + x.second
        }
    }

    operator fun get(vararg indices: Int): Float {
        return elements[index(indices)]
    }

    operator fun get(vararg ranges: IntRange): Tensor {
        val size = ranges.size
        val shape = ranges.mapToIntArray { x -> x.endInclusive - x.start + 1 }
        val reversedShape = shape.reversed()
        val indices = IntArray(size)
        val elements = FloatArray(shape.fold(1, Int::times)) {
            var i = it
            var dimensionIndex = size - 1
            for (dimension in reversedShape) {
                indices[dimensionIndex] = i % dimension + ranges[dimensionIndex].start
                i /= dimension
                dimensionIndex--
            }
            get(*indices)
        }
        return Tensor(Shape(*shape), elements)
    }

    override fun equals(other: Any?): Boolean {
        if (other !is Tensor) { return false }

        return shape == other.shape && zipFold(elements, other.elements, true) { result, lhs, rhs ->
            if (!result) { return false }
            lhs == rhs
        }
    }

    private inline fun commutativeBinaryOperation(tensor: Tensor, operation: (Float, Float) -> Float): Tensor {
        val lSize = shape.dimensions.size
        val rSize = tensor.shape.dimensions.size

        if (lSize == rSize) {
            assert({ shape == tensor.shape }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return Tensor(shape, zipMap(elements, tensor.elements, operation))
        }

        val a: Tensor
        val b: Tensor
        if (lSize < rSize) {
            a = tensor
            b = this
        } else {
            a = this
            b = tensor
        }
        assert({ a.shape.dimensions.endsWith(b.shape.dimensions) }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })

        return Tensor(a.shape, zipMapRepeat(a.elements, b.elements, operation))
    }

    private inline fun noncommutativeBinaryOperation(tensor: Tensor, operation: (Float, Float) -> Float, reverseOperation: (Float, Float) -> Float): Tensor {
        val lSize = shape.dimensions.size
        val rSize = tensor.shape.dimensions.size

        if (lSize == rSize) {
            assert({ shape == tensor.shape }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return Tensor(shape, zipMap(elements, tensor.elements, operation))
        } else if (lSize < rSize) {
            assert({ tensor.shape.dimensions.endsWith(shape.dimensions) }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return Tensor(tensor.shape, zipMapRepeat(tensor.elements, elements, reverseOperation))
        } else {
            assert({ shape.dimensions.endsWith(tensor.shape.dimensions) }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return Tensor(shape, zipMapRepeat(elements, tensor.elements, operation))
        }
    }

    operator fun plus(tensor: Tensor): Tensor {
        return commutativeBinaryOperation(tensor) { lhs, rhs -> lhs + rhs }
    }

    operator fun minus(tensor: Tensor): Tensor {
        return noncommutativeBinaryOperation(tensor, { lhs, rhs -> lhs - rhs }, { lhs, rhs -> rhs - lhs})
    }

    operator fun times(tensor: Tensor): Tensor {
        return commutativeBinaryOperation(tensor) { lhs, rhs -> lhs * rhs }
    }

    operator fun div(tensor: Tensor): Tensor {
        return noncommutativeBinaryOperation(tensor, { lhs, rhs -> lhs / rhs }, { lhs, rhs -> rhs / lhs})
    }

    operator fun times(scalar: Float): Tensor {
        return Tensor(shape, elements.map { it * scalar })
    }

    operator fun div(scalar: Float): Tensor {
        return Tensor(shape, elements.map { it / scalar })
    }

    fun matmul(tensor: Tensor): Tensor {
        assert({ shape.dimensions.size == 2 }, { "This tensor is not a matrix: shape = ${shape}" })
        assert({ tensor.shape.dimensions.size == 2 }, { "The given tensor is not a matrix: shape = ${tensor.shape}" })

        val inCols1Rows2 = shape.dimensions[1]
        assert({ tensor.shape.dimensions[0] == inCols1Rows2 }, { "Incompatible shapes of matrices: self.shape = ${shape}, tensor.shape = ${tensor.shape}" })

        val outRows = shape.dimensions[0]
        val outCols = tensor.shape.dimensions[1]

        var elements = FloatArray(outRows * outCols)
        for (r in 0 until outRows) {
            for (i in 0 until inCols1Rows2) {
                var elementIndex = r * outCols
                val left = this.elements[r * inCols1Rows2 + i]
                for (c in 0 until outCols) {
                    elements[elementIndex] += left * tensor.elements[i * outCols + c]
                    elementIndex++
                }
            }
        }

        return Tensor(Shape(outRows, outCols), elements)
    }

    override fun toString(): String {
        return "Tensor(${shape}, ${Arrays.toString(elements)})"
    }
}

operator fun Float.times(tensor: Tensor): Tensor {
    return tensor.times(this)
}

