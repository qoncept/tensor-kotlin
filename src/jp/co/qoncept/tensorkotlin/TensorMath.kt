package jp.co.qoncept.tensorkotlin

fun Tensor.pow(tensor: Tensor):Tensor {
    assert({ shape == tensor.shape }, { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
    return Tensor(shape, zipMap(elements, tensor.elements) { a, b -> Math.pow(a.toDouble(), b.toDouble()).toFloat() })
}

fun Tensor.pow(scalar: Float): Tensor {
    return Tensor(shape,elements.map { Math.pow(it.toDouble(), scalar.toDouble()).toFloat() })
}

val Tensor.sin: Tensor
    get() = Tensor(shape, elements.map { Math.sin(it.toDouble()).toFloat() })

val Tensor.cos: Tensor
    get() = Tensor(shape, elements.map { Math.cos(it.toDouble()).toFloat() })

val Tensor.tan: Tensor
    get() = Tensor(shape, elements.map { Math.tan(it.toDouble()).toFloat() })

val Tensor.asin: Tensor
    get() = Tensor(shape, elements.map { Math.asin(it.toDouble()).toFloat() })

val Tensor.acos: Tensor
    get() = Tensor(shape, elements.map { Math.acos(it.toDouble()).toFloat() })

val Tensor.atan: Tensor
    get() = Tensor(shape, elements.map { Math.atan(it.toDouble()).toFloat() })

val Tensor.sinh: Tensor
    get() = Tensor(shape, elements.map { Math.sinh(it.toDouble()).toFloat() })

val Tensor.cosh: Tensor
    get() = Tensor(shape, elements.map { Math.cosh(it.toDouble()).toFloat() })

val Tensor.tanh: Tensor
    get() = Tensor(shape, elements.map { Math.tanh(it.toDouble()).toFloat() })

val Tensor.exp: Tensor
    get() = Tensor(shape, elements.map { Math.exp(it.toDouble()).toFloat() })

val Tensor.log: Tensor
    get() = Tensor(shape, elements.map { Math.log(it.toDouble()).toFloat() })

val Tensor.sqrt: Tensor
    get() = Tensor(shape, elements.map { Math.sqrt(it.toDouble()).toFloat() })

val Tensor.cbrt: Tensor
    get() = Tensor(shape, elements.map { Math.cbrt(it.toDouble()).toFloat() })

val Tensor.sigmoid: Tensor
    get() = Tensor(shape, elements.map { (1.0 / Math.exp(-it.toDouble())).toFloat() })
