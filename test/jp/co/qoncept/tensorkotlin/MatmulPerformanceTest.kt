package jp.co.qoncept.tensorkotlin

import org.junit.Test

private fun createTensor1000x1000(): Tensor {
    val elements = floatArrayOf(1000 * 1000, 0.1f)
    return Tensor(Shape(1000, 1000), elements)
}

private fun createTensor1x1000(): Tensor {
    val elements = floatArrayOf(1 * 1000, 0.1f)
    return Tensor(Shape(1, 1000), elements)
}

class MatmulPerformanceTest {
    @Test
    fun testMultiplication(){
        val W = createTensor1000x1000()
        val x = createTensor1x1000()

        measureBlock {
            x.matmul(W)
        }
    }
}