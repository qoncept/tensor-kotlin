package jp.co.qoncept.tensorkotlin

import org.junit.Assert.*
import org.junit.Test

class TensorNNTest {
    @Test
    fun testMaxPool() {
        run {
            val a = Tensor(Shape(2,3,1), floatArrayOf(0,1,2,3,4,5))
            var r = a.maxPool(intArrayOf(1,3,1), intArrayOf(1,1,1))
            assertEquals(Tensor(Shape(2,3,1), floatArrayOf(1,2,2,4,5,5)), r)
        }

        run {
            val b = Tensor(Shape(2,2,2), floatArrayOf(0,1,2,3,4,5,6,7))

            run {
                val r = b.maxPool(intArrayOf(1,2,1), intArrayOf(1,1,1))
                assertEquals(Tensor(Shape(2,2,2), floatArrayOf(2, 3, 2, 3, 6, 7, 6, 7)), r)
            }

            run {
                val r = b.maxPool(intArrayOf(1,2,1), intArrayOf(1,2,1))
                assertEquals(Tensor(Shape(2,1,2), floatArrayOf(2, 3, 6, 7)), r)
            }
        }
    }

    @Test
    fun testConv2d() {
        run {
            val a = Tensor(Shape(2,4,1), floatArrayOf(1,2,3,4,5,6,7,8))
            run {
                val filter = Tensor(Shape(2,1,1,2), floatArrayOf(1,2,1,2))
                val result = a.conv2d(filter, intArrayOf(1,1,1))
                assertEquals(Tensor(Shape(2,4,2), floatArrayOf(6,12,8,16,10,20,12,24,5,10,6,12,7,14,8,16)), result)
            }

            run {
                val filter = Tensor(Shape(1,1,1,5), floatArrayOf(1,2,1,2,3))
                val result = a.conv2d(filter, intArrayOf(1,1,1))
                assertEquals(Tensor(Shape(2,4,5), floatArrayOf(1,2,1,2,3,2,4,2,4,6,3,6,3,6,9,4,8,4,8,12,5,10,5,10,15,6,12,6,12,18,7,14,7,14,21,8,16,8,16,24)), result)
            }
        }

        run {
            val a = Tensor(Shape(2,2,4), floatArrayOf(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
            val filter = Tensor(Shape(1,1,4,2), floatArrayOf(1,2,1,2,3,2,1,1))
            val result = a.conv2d(filter, intArrayOf(1,1,1))
            assertEquals(Tensor(Shape(2,2,2), floatArrayOf(16, 16, 40, 44, 64, 72, 88, 100)), result)
        }

        run {
            val a = Tensor(Shape(4,2,2), floatArrayOf(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
            val filter = Tensor(Shape(2,2,2,1), floatArrayOf(1,2,1,2,3,2,1,1))
            val result = a.conv2d(filter, intArrayOf(2,2,1))
            assertEquals(Tensor(Shape(2,1,1), floatArrayOf(58,162)), result)
        }

        run {
            val a = Tensor(Shape(4,4,1), floatArrayOf(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
            val filter = Tensor(Shape(3,3,1,1), floatArrayOf(1,2,1,2,3,2,1,1,1))
            val result = a.conv2d(filter, intArrayOf(3,3,1))
            assertEquals(Tensor(Shape(2,2,1), floatArrayOf(18,33,95,113)), result)
        }

        run {
            val a = Tensor(Shape(10, 10, 1), naturalNumbers(10 * 10))
            val filter = Tensor(Shape(5, 5, 1, 32), naturalNumbers(5 * 5 * 32))
            val result = a.conv2d(filter, intArrayOf(1, 1, 1))
            assertEquals(Shape(10, 10, 32),  result.shape)
            assertEquals(66816.0f, result[0, 0, 0])
            assertEquals(66915.0f, result[0, 0, 1])
            assertEquals(67014.0f, result[0, 0, 2])
            assertEquals(69687.0f, result[0, 0, 29])
            assertEquals(69786.0f, result[0, 0, 30])
            assertEquals(69885.0f, result[0, 0, 31])
            assertEquals(90560.0f, result[0, 1, 0])
            assertEquals(114880.0f, result[0, 2, 0])
            assertEquals(155680.0f, result[0, 7, 0])
            assertEquals(124160.0f, result[0, 8, 0])
            assertEquals(92736.0f, result[0, 9, 0])
            assertEquals(184824.0f, result[9, 9, 29])
            assertEquals(185616.0f, result[9, 9, 30])
            assertEquals(186408.0f, result[9, 9, 31])
        }
    }

    @Test
    fun testMatmuladd() {
        run {
            val a = floatArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
            val b = floatArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
            val out = FloatArray(15)
            matmuladd(3, 2, 2, a, 5, b, 7, out)
            /*
                         | 5  6 |
                | 2 3 4| | 7  8 |
                         | 9 10 |
             */
            assertEquals(floatArrayOf(0, 0, 0, 0, 0, 0, 0, 67, 76, 0, 0, 0, 0, 0, 0).toList(), out.toList())
        }

        run {
            val a = floatArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
            val b = floatArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
            val out = FloatArray(15)
            matmuladd(3, 2, 2, a, 5, b, 7, out)
            matmuladd(3, 2, 2, a, 5, b, 7, out)
            /*
                         | 5  6 |            | 5  6 |
                | 2 3 4| | 7  8 | + | 2 3 4| | 7  8 |
                         | 9 10 |            | 9 10 |
             */
            assertEquals(floatArrayOf(0, 0, 0, 0, 0, 0, 0, 134, 152, 0, 0, 0, 0, 0, 0).toList(), out.toList())
        }
    }
}