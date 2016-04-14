package jp.co.qoncept.tensorkotlin

val Tensor.softmax: Tensor
    get() {
        val exps = exp
        val sum = exps._elements.fold(0.0f) { r, x -> r + x }
        return exps / sum
    }

val Tensor.relu: Tensor
    get() {
        return Tensor(shape, _elements.map { Math.max(it, 0.0f) })
    }
