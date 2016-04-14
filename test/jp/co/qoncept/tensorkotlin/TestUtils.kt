package jp.co.qoncept.tensorkotlin


internal fun floatArrayOf(vararg elements: Int): FloatArray {
    val result = FloatArray(elements.size)
    for (i in elements.indices) {
        result[i] = elements[i].toFloat()
    }
    return result
}

internal fun measureBlock(procedure: () -> Unit) {
    run {
        val stackTrace = Thread.getAllStackTraces().values.iterator().next()
        val iterator = stackTrace.iterator()
        iterator.next()
        val element = iterator.next()
        println("measureBlock: ${element.className}\$${element.methodName}")
    }

    val N = 10
    var total: Long = 0
    for (i in 1..N) {
        val begin = System.currentTimeMillis()
        procedure()
        val end = System.currentTimeMillis()
        val elapsed = end - begin

        println("${i}: ${elapsed / 1000.0} [s]")

        total += elapsed
    }

    println("avg: ${total / 1000.0 / N} [s]")
    println()
}