package jp.co.qoncept.tensorkotlin

val Tensor.softmax: Tensor
    get() {
        val exps = exp
        val sum = exps.elements.fold(0.0f) { r, x -> r + x }
        return exps / sum
    }

val Tensor.relu: Tensor
    get() {
        return Tensor(shape, this.elements.map { Math.max(it, 0.0f) })
    }

fun Tensor.maxPool(kernelSize: IntArray, strides: IntArray): Tensor {
    assert({ shape.dimensions.size == 3 }, { "`shape.dimensions.size` must be 3: ${shape.dimensions.size}" })
    assert({ kernelSize.size == 3 }, { "`kernelSize.size` must be 3: ${kernelSize.size}" })
    assert({ kernelSize[2] == 1 } , { "`kernelSize[2]` != 1 is not supported: ${ kernelSize[2] }" })
    assert({ strides.size == 3 }, { "`strides.size` must be 3: ${ strides.size }" })
    assert({ strides[2] == 1 } , { "`strides[2]` != 1 is not supported: ${ strides[2] }" })

    val inRows = shape.dimensions[0]
    val inCols = shape.dimensions[1]
    val numChannels = shape.dimensions[2]

    val filterHeight = kernelSize[0]
    val filterWidth = kernelSize[1]

    val inMinDy = -(filterHeight - 1) / 2
    val inMaxDy = inMinDy + filterHeight - 1
    val inMinDx = -(filterWidth - 1) / 2
    val inMaxDx = inMinDx + filterWidth - 1

    val rowStride = strides[0]
    val colStride = strides[1]

    val outRows = inRows ceilDiv rowStride
    val outCols = inCols ceilDiv colStride

    val elements = FloatArray(outCols * outRows * numChannels)

    var elementIndex = 0
    for (y in 0 until outRows) {
        val inY0 = y * rowStride
        val inMinY = Math.max(inY0 + inMinDy, 0)
        val inMaxY = Math.min(inY0 + inMaxDy, inRows - 1)

        for (x in 0 until outCols) {
            val inX0 = x * colStride
            val inMinX = Math.max(inX0 + inMinDx, 0)
            val inMaxX = Math.min(inX0 + inMaxDx, inCols - 1)

            for (c in 0 until numChannels) {
                var maxElement = Float.MIN_VALUE
                for (inY in inMinY..inMaxY) {
                    for (inX in inMinX..inMaxX) {
                        maxElement = Math.max(maxElement, this.elements[(inY * inCols + inX) * numChannels + c])
                    }
                }
                elements[elementIndex++] = maxElement
            }
        }
    }

    return Tensor(Shape(outRows, outCols, numChannels), elements)
}

fun Tensor.conv2d(filter: Tensor, strides: IntArray): Tensor {
    val inChannels = filter.shape.dimensions[2]

    assert({ shape.dimensions.size == 3 }, { "`shape.dimensions.size` must be 3: ${shape.dimensions.size}" })
    assert({ filter.shape.dimensions.size == 4 }, { "`filter.shape.dimensions.size` must be 4: ${filter.shape.dimensions.size}" })
    assert({ strides.size == 3 }, { "`strides.size` must be 3: ${ strides.size }" })
    assert({ strides[2] == 1 } , { "`strides[2]` != 1 is not supported: ${ strides[2] }" })
    assert({ shape.dimensions[2] == inChannels }, { "The number of channels of this tensor and the filter are not compatible: ${shape.dimensions[2]} != ${inChannels}" })

    val inRows = shape.dimensions[0]
    val inCols = shape.dimensions[1]

    val filterHeight = filter.shape.dimensions[0]
    val filterWidth = filter.shape.dimensions[1]

    val inMinDy = -(filterHeight - 1) / 2
    val inMaxDy = inMinDy + filterHeight - 1
    val inMinDx = -(filterWidth - 1) / 2
    val inMaxDx = inMinDx + filterWidth - 1

    val rowStride = strides[0]
    val colStride = strides[1]

    val outRows = shape.dimensions[0] ceilDiv rowStride
    val outCols = shape.dimensions[1] ceilDiv colStride
    val outChannels = filter.shape.dimensions[3]

    val elements = FloatArray(outCols * outRows * outChannels)

    for (y in 0 until outRows) {
        val inY0 = y * rowStride
        val inMinY = Math.max(inY0 + inMinDy, 0)
        val inMaxY = Math.min(inY0 + inMaxDy, inRows - 1)

        for (x in 0 until outCols) {
            val inX0 = x * colStride
            val inMinX = Math.max(inX0 + inMinDx, 0)
            val inMaxX = Math.min(inX0 + inMaxDx, inCols - 1)

            val inYOffset = inY0 + inMinDy
            val inXOffset = inX0 + inMinDx

            for (inY in inMinY..inMaxY) {
                for (inX in inMinX..inMaxX) {
                    matmuladd(
                            inChannels, outChannels,
                            (inY * inCols + inX) * inChannels, this.elements,
                            ((inY - inYOffset) * filterWidth + (inX - inXOffset)) * inChannels * outChannels, filter.elements,
                            (y * outCols + x) * outChannels, elements
                    )
                }
            }
        }
    }

    return Tensor(Shape(outRows, outCols, outChannels), elements)
}

internal fun matmuladd(inCols1Rows2: Int, outCols: Int, o1: Int, vec: FloatArray, o2: Int, mat: FloatArray, oo: Int, out: FloatArray) {
    for (i in 0 until inCols1Rows2) {
        var elementIndex = oo
        val left = vec[i + o1]
        for (c in 0 until outCols) {
            out[elementIndex] += left * mat[i * outCols + c + o2]
            elementIndex++
        }
    }
}
