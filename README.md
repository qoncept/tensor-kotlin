# TensorKotlin

_TensorKotlin_ is a lightweight library to calculate tensors, which has similar APIs to [_TensorFlow_](https://www.tensorflow.org/)'s. _TensorKotlin_ is useful to simulate calculating tensors in Kotlin __using models trained by _TensorFlow___.

```kotlin
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
```

## License

[The MIT License](LICENSE)
