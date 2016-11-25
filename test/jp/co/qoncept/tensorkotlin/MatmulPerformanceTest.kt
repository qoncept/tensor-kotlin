package jp.co.qoncept.tensorkotlin

import org.testng.annotations.Test

private val dimension = 10000

private fun createMatrix(): Tensor {
    val elements = FloatArray(dimension * dimension).map { Math.random().toFloat() }
    return Tensor(Shape(dimension, dimension), elements)
}

private fun createVector(): Tensor {
    val elements = FloatArray(1 * dimension).map { Math.random().toFloat() }
    return Tensor(Shape(1, dimension), elements)
}

class MatmulPerformanceTest {
    @Test
    fun testMatmul(){
        val W = createMatrix()
        val x = createVector()

        measureBlock {
            x.matmul(W)
        }
    }
}