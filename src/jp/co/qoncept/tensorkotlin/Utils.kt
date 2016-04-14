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