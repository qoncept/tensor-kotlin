package jp.co.qoncept.tensorkotlin

import org.testng.Assert.*
import org.testng.annotations.Test

class TensorKotlinSample {
    @Test
    fun testSample() {
        val a = Tensor(Shape(2, 3), floatArrayOf(1, 2, 3, 4, 5, 6)) // [[1, 2, 3], [4, 5, 6]]
        val b = Tensor(Shape(2, 3), floatArrayOf(7, 8, 9, 10, 11, 12)) // [[7, 8, 9], 10, 11, 12]]

        val x = a[1, 2] // 6.0f
        val sub = a[0..1, 1..2] // [[2, 3], [5, 6]]

        val sum = a + b // [[8, 10, 12], [14, 16, 18]]
        val mul = a * b // [[7, 16, 27], [40, 55, 72]]

        val c = Tensor(Shape(3, 1), floatArrayOf(7, 8, 9)) // [[7], [8], [9]]
        val matmul = a.matmul(c) // [[50], [122]]

        val zeros = Tensor(Shape(2, 3, 4))
        val ones = Tensor(Shape(2, 3, 4), 1.0f)

        assertEquals(x, 6.0f)
        assertEquals(sub, Tensor(Shape(2, 2), floatArrayOf(2, 3, 5, 6)))
        assertEquals(sum, Tensor(Shape(2, 3), floatArrayOf(8, 10, 12, 14, 16, 18)))
        assertEquals(mul, Tensor(Shape(2, 3), floatArrayOf(7, 16, 27, 40, 55, 72)))
        assertEquals(matmul, Tensor(Shape(2, 1), floatArrayOf(50, 122)))
        assertEquals(zeros, Tensor(Shape(2, 3, 4), floatArrayOf(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        assertEquals(ones, Tensor(Shape(2, 3, 4), floatArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)))
    }
}
