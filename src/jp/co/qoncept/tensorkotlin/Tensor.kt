package jp.co.qoncept.tensorkotlin

class Tensor(val shape: Shape, elements: FloatArray) {
    private val _elements = elements

    val elements: FloatArray
        get() = _elements.clone()

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

    override fun equals(other: Any?): Boolean {
        if (other !is Tensor) { return false }

        return shape == other.shape && zipFold(elements, other.elements, true) { result, lhs, rhs ->
            if (!result) { return false }
            lhs == rhs
        }
    }

    operator fun plus(tensor: Tensor): Tensor {
        assert({ shape == tensor.shape }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
        return Tensor(shape, zipMap(elements, tensor.elements) { lhs, rhs -> lhs + rhs })
    }

    operator fun minus(tensor: Tensor): Tensor {
        assert({ shape == tensor.shape }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
        return Tensor(shape, zipMap(elements, tensor.elements) { lhs, rhs -> lhs - rhs })
    }

    operator fun times(tensor: Tensor): Tensor {
        assert({ shape == tensor.shape }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
        return Tensor(shape, zipMap(elements, tensor.elements) { lhs, rhs -> lhs * rhs })
    }

    operator fun div(tensor: Tensor): Tensor {
        assert({ shape == tensor.shape }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
        return Tensor(shape, zipMap(elements, tensor.elements) { lhs, rhs -> lhs / rhs })
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

        val n = shape.dimensions[1]
        assert({ tensor.shape.dimensions[0] == n }, { "Incompatible shapes of matrices: self.shape = ${shape}, tensor.shape = ${tensor.shape}" })

        val numRows = shape.dimensions[0]
        val numCols = tensor.shape.dimensions[1]

        var elements = FloatArray(numRows * numCols)
        for (r in 0 until numRows) {
            for (i in 0 until n) {
                var elementIndex = r * numCols
                val left = _elements[r * n + i]
                for (c in 0 until numCols) {
                    elements[elementIndex] += left * tensor._elements[i * numCols + c]
                    elementIndex++
                }
            }
        }

        return Tensor(Shape(numRows, numCols), elements)
    }
}

operator fun Float.times(tensor: Tensor): Tensor {
    return tensor.times(this)
}

inline private fun assert(value: () -> Boolean, lazyMessage: () -> Any) {
    assert(Tensor::class.java, value, lazyMessage)
}
