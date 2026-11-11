import tf from '@tensorflow/tfjs'

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential()

  model.add(
    tf.layers.dense({ inputShape: [7], units: 300, activation: 'relu' }),
  )
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  await model.fit(inputXs, outputYs, {
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`)
      },
    },
  })

  return model
}

async function prediction(model, pessoa) {
  const tensorPessoa = tf.tensor2d(pessoa)

  const prediction = model.predict(tensorPessoa)
  const predictionArray = await prediction.array()

  return predictionArray[0].map((prob, index) => ({ prob, index }))
}

const pessoas = [
  { nome: 'Carlito', idade: 20, sexo: 'M', moto: 'elétrica' },
  { nome: 'Bruninho', idade: 20, sexo: 'M', moto: 'cg' },
  { nome: 'Joana', idade: 25, sexo: 'F', moto: 'biz' },
  { nome: 'Pedro', idade: 30, sexo: 'M', moto: 'cg' },
  { nome: 'Jaqueline', idade: 38, sexo: 'F', moto: 'biz' },
  { nome: 'Frederico', idade: 44, sexo: 'M', moto: 'vstrom' },
]

// Ordem: [idade_normalizada, sexo_masculino, sexo_feminino, moto_elétrica, moto_biz, moto_cg, moto_vstrom]
// Função para normalizar a idade
const normalizarIdade = idade => (idade - 20) / (44 - 20)

const tensorPessoasNormalizadas = [
  [normalizarIdade(20), 1, 0, 1, 0, 0, 0], // Carlito
  [normalizarIdade(20), 1, 0, 0, 0, 1, 0], // Bruninho
  [normalizarIdade(25), 0, 1, 0, 1, 0, 0], // Joana
  [normalizarIdade(30), 1, 0, 0, 0, 1, 0], // Pedro
  [normalizarIdade(38), 0, 1, 0, 1, 0, 0], // Jaqueline
  [normalizarIdade(44), 1, 0, 0, 0, 0, 1], // Frederico
]

const categorias = ['iniciante', 'intermediario', 'experiente']
const tensorCategorias = [
  [1, 0, 0], // iniciante - Carlito
  [1, 0, 0], // iniciante - Bruninho
  [1, 0, 0], // iniciante - Joana
  [0, 1, 0], // intermediario - Pedro
  [0, 0, 1], // intermediario - Jaqueline
  [0, 0, 1], // experiente - Frederico
]

const inputXs = tf.tensor2d(tensorPessoasNormalizadas)
const outputYs = tf.tensor2d(tensorCategorias)

// inputXs.print()
// outputYs.print()

const model = await trainModel(inputXs, outputYs)

const novaPessoa = { nome: 'Lucas', idade: 20, sexo: 'M', moto: 'cg' }
const novaPessoaTensorNormalizada = [
  [
    normalizarIdade(novaPessoa.idade),
    novaPessoa.sexo === 'M' ? 1 : 0,
    novaPessoa.sexo === 'F' ? 1 : 0,
    novaPessoa.moto === 'elétrica' ? 1 : 0,
    novaPessoa.moto === 'biz' ? 1 : 0,
    novaPessoa.moto === 'cg' ? 1 : 0,
    novaPessoa.moto === 'vstrom' ? 1 : 0,
  ],
]

console.log('\n Pessoa para previsão')
console.log(novaPessoa)
console.log(novaPessoaTensorNormalizada)

console.log('\nResultado:')
const previsao = await prediction(model, novaPessoaTensorNormalizada)
// console.log(previsao)

const resultado = previsao
  .sort((a, b) => b.prob - a.prob)
  .map(
    p =>
      `Categoria: ${categorias[p.index]}, Probabilidade: ${(p.prob * 100).toFixed(2)}%`,
  )
  .join('\n')

console.log(resultado)
