package jp.co.qoncept.tensorkotlin

internal fun floatArrayOf(size: Int, repeatedValue: Float): FloatArray {
    val array = FloatArray(size)
    array.fill(repeatedValue)
    return array
}

internal inline fun zipMap(a: FloatArray, b: FloatArray, operation: (Float, Float) -> Float): FloatArray {
    val result = FloatArray(a.size)
    for (i in a.indices) {
        result[i] = operation(a[i], b[i])
    }
    return result
}

internal inline fun zipMapRepeat(a: FloatArray, infiniteB: FloatArray, operation: (Float, Float) -> Float): FloatArray {
    val result = FloatArray(a.size)
    for (i in a.indices) {
        result[i] = operation(a[i], infiniteB[i % infiniteB.size])
    }
    return result
}

internal inline fun <R> zipFold(a: FloatArray, b: FloatArray, initial: R, operation: (R, Float, Float) -> R): R {
    var result: R = initial
    for (i in a.indices) {
        result = operation(result, a[i], b[i])
    }
    return result
}

internal inline fun <R> zipFold(a: IntArray, b: IntArray, initial: R, operation: (R, Int, Int) -> R): R {
    var result: R = initial
    for (i in a.indices) {
        result = operation(result, a[i], b[i])
    }
    return result
}

internal inline fun FloatArray.map(transform: (Float) -> Float): FloatArray {
    val result = FloatArray(size)
    for (i in indices) {
        result[i] = transform(this[i])
    }
    return result
}

internal inline fun <T> Array<out T>.map(transform: (T) -> Int): IntArray {
    val result = IntArray(size)
    for (i in indices) {
        result[i] = transform(this[i])
    }
    return result
}

internal fun IntArray.endsWith(suffix: IntArray): Boolean {
    if (size < suffix.size) { return false }
    val offset = size - suffix.size
    for (i in suffix.indices) {
        if (this[offset + i] != suffix[i]) {
            return false
        }
    }
    return true
}

internal inline fun assert(value: () -> Boolean, lazyMessage: () -> Any) {
    if (Tensor::class.java.desiredAssertionStatus()) {
        if (!value()) {
            val message = lazyMessage()
            throw AssertionError(message)
        }
    }
}

internal infix fun Int.ceilDiv(rhs: Int): Int {
    return (this + rhs - 1) / rhs
}