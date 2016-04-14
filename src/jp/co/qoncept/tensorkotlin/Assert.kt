package jp.co.qoncept.tensorkotlin

internal inline fun <T> assert(clazz: Class<T>, value: () -> Boolean, lazyMessage: () -> Any) {
    if (clazz.desiredAssertionStatus()) {
        if (!value()) {
            val message = lazyMessage()
            throw AssertionError(message)
        }
    }
}
