// Importa a biblioteca TensorFlow.js
const tf = require('@tensorflow/tfjs-node');

// Cria um modelo sequencial
const model = tf.sequential();

// Adiciona uma camada densa ao modelo
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Compila o modelo com um otimizador e uma função de perda
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Dados de treinamento
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([2, 4, 6, 8], [4, 1]);

// Treina o modelo
model.fit(xs, ys, { epochs: 500 }).then(() => {
  // Faz uma previsão usando o modelo treinado
  const prediction = model.predict(tf.tensor2d([5], [1, 1]));
  prediction.print();
});
