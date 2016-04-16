package jp.co.qoncept.tensorkotlin

import org.junit.Assert.*
import org.junit.Test

class TensorTest {
    @Test
    fun testIndex() {
        run {
            val a = Tensor(Shape())
            assertEquals(0, a.index(intArrayOf()))
        }

        run {
            val a = Tensor(Shape(7))
            assertEquals(3, a.index(intArrayOf(3)))
        }

        run {
            val a = Tensor(Shape(5, 7))
            assertEquals(9, a.index(intArrayOf(1, 2)))
        }

        run {
            val a = Tensor(Shape(5, 7, 11))
            assertEquals(244, a.index(intArrayOf(3, 1, 2)))
        }
    }

    @Test
    fun testAdd() {
        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(1, 2, 3, 4, 5, 6))
            val b = Tensor(Shape(2, 3), floatArrayOf(7, 8, 9, 10, 11, 12))
            val r = a + b
            assertEquals(Tensor(Shape(2, 3), floatArrayOf(8, 10, 12, 14, 16, 18)), r)
        }
    }

    @Test
    fun testSub() {
        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(1, 2, 3, 4, 5, 6))
            val b = Tensor(Shape(2, 3), floatArrayOf(12, 11, 10, 9, 8, 7))
            val r = a - b
            assertEquals(Tensor(Shape(2, 3), floatArrayOf(-11, -9, -7, -5, -3, -1)), r)
        }
    }

    @Test
    fun testMul() {
        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(1, 2, 3, 4, 5, 6))
            val b = Tensor(Shape(2, 3), floatArrayOf(7, 8, 9, 10, 11, 12))
            val r = a * b
            assertEquals(Tensor(Shape(2, 3), floatArrayOf(7, 16, 27, 40, 55, 72)), r)
        }
    }

    @Test
    fun testDiv() {
        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(1, 2, 3, 4, 5, 6))
            val b = Tensor(Shape(2, 3), floatArrayOf(2, 4, 8, 16, 32, 64))
            val r = a / b
            assertEquals(Tensor(Shape(2, 3), floatArrayOf(0.5f, 0.5f, 0.375f, 0.25f, 0.15625f, 0.09375f)), r)
        }
    }

    @Test
    fun testMatmul() {
        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(1, 2, 3, 4, 5, 6))
            val b = Tensor(Shape(3, 4), floatArrayOf(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
            val r = a.matmul(b)
            assertEquals(Tensor(Shape(2, 4), floatArrayOf(74, 80, 86, 92, 173, 188, 203, 218)), r)
        }
    }

    @Test
    fun testEquals() {
        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(2, 3, 5, 7, 11, 13))
            val b = Tensor(Shape(2, 3), floatArrayOf(2, 3, 5, 7, 11, 13))
            assertTrue(a == b)
        }

        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(2, 3, 5, 7, 11, 13))
            val b = Tensor(Shape(2, 3), floatArrayOf(2, 3, 5, 7, 11, 17))
            assertFalse(a == b)
        }

        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(2, 3, 5, 7, 11, 13))
            val b = Tensor(Shape(3, 2), floatArrayOf(2, 3, 5, 7, 11, 17))
            assertFalse(a == b)
        }

        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(2, 3, 5, 7, 11, 13))
            val b = Tensor(Shape(2, 2), floatArrayOf(2, 3, 5, 7))
            assertFalse(a == b)
        }

        run {
            val a = Tensor(Shape(2, 3), floatArrayOf(2, 3, 5, 7, 11, 13))
            val b = Tensor(Shape(), floatArrayOf())
            assertFalse(a == b)
        }

        run {
            val a = Tensor(Shape(), floatArrayOf())
            val b = Tensor(Shape(), floatArrayOf())
            assertTrue(a == b)
        }
    }
}