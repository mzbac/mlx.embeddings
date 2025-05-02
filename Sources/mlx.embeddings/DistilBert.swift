public struct DistilBertConfiguration: Decodable, Sendable {
  var layer_norm_eps: Float = 1e-12
  var max_trained_positions: Int = 2048
  var hidden_size: Int = 768
  var num_attention_heads: Int = 12
  var intermediate_size: Int = 3072
  var num_hidden_layers: Int = 12
  var vocab_size: Int = 30528
  var max_position_embeddings: Int = 0
  var model_type: String

  enum CodingKeys: String, CodingKey {
    case layer_norm_eps
    case max_trained_positions
    case vocab_size
    case max_position_embeddings
    case model_type
  }

  enum DistilBertCodingKeys: String, CodingKey {
    case hidden_size = "dim"
    case num_hidden_layers = "n_layers"
    case num_attention_heads = "n_heads"
    case intermediate_size = "hidden_dim"
  }

  public init(from decoder: Decoder) throws {
    let container: KeyedDecodingContainer<CodingKeys> =
      try decoder.container(
        keyedBy: CodingKeys.self)
    layer_norm_eps =
      try container.decodeIfPresent(
        Float.self,
        forKey: CodingKeys.layer_norm_eps)
      ?? 1e-12
    max_trained_positions =
      try container.decodeIfPresent(
        Int.self,
        forKey: CodingKeys.max_trained_positions) ?? 2048
    vocab_size =
      try container.decodeIfPresent(
        Int.self,
        forKey: CodingKeys.vocab_size)
      ?? 30528
    max_position_embeddings =
      try container.decodeIfPresent(
        Int.self,
        forKey: CodingKeys.max_position_embeddings) ?? 0
    model_type = try container.decode(String.self, forKey: CodingKeys.model_type)
    let distilBertConfig: KeyedDecodingContainer<DistilBertCodingKeys> =
      try decoder.container(
        keyedBy: DistilBertCodingKeys.self)

    hidden_size =
      try distilBertConfig.decodeIfPresent(
        Int.self,
        forKey: DistilBertCodingKeys.hidden_size.self) ?? 768
    num_attention_heads =
      try distilBertConfig.decodeIfPresent(
        Int.self,
        forKey: DistilBertCodingKeys.num_attention_heads.self) ?? 12
    intermediate_size =
      try distilBertConfig.decodeIfPresent(
        Int.self, forKey: DistilBertCodingKeys.intermediate_size.self)
      ?? 3072
    num_hidden_layers =
      try distilBertConfig.decodeIfPresent(
        Int.self,
        forKey: DistilBertCodingKeys.num_hidden_layers.self) ?? 12
  }
}

import Foundation
import MLX
import MLXFast
import MLXNN


private class BertEmbedding: Module {

  @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
  @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
  @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding

  init(_ config: DistilBertConfiguration) {
    _wordEmbeddings.wrappedValue = Embedding(
      embeddingCount: config.vocab_size, dimensions: config.hidden_size)
    _norm.wrappedValue = LayerNorm(
      dimensions: config.hidden_size, eps: config.layer_norm_eps)

    _positionEmbeddings.wrappedValue = Embedding(
      embeddingCount: config.max_position_embeddings,
      dimensions: config.hidden_size)

  }

  func callAsFunction(
    inputIds: MLXArray,
    positionIds: MLXArray? = nil,
    tokenTypeIds: MLXArray? = nil
  ) -> MLXArray {
    let posIds = positionIds ?? broadcast(MLXArray.arange(inputIds.dim(1)), to: inputIds.shape)
    let words = wordEmbeddings(inputIds) + positionEmbeddings(posIds)

    return norm(words)
  }
}

private class BertAttention: Module {
  let numAttentionHeads: Int
  let attentionHeadSize: Int
  let allHeadSize: Int

  @ModuleInfo(key: "q_lin") var query: Linear
  @ModuleInfo(key: "k_lin") var key: Linear
  @ModuleInfo(key: "v_lin") var value: Linear
  @ModuleInfo(key: "out_lin") var out_lin: Linear

  init(_ config: DistilBertConfiguration) {
    numAttentionHeads = config.num_attention_heads
    attentionHeadSize = config.hidden_size / config.num_attention_heads
    allHeadSize = numAttentionHeads * attentionHeadSize

    _query.wrappedValue = Linear(config.hidden_size, allHeadSize)
    _key.wrappedValue = Linear(config.hidden_size, allHeadSize)
    _value.wrappedValue = Linear(config.hidden_size, allHeadSize)
    _out_lin.wrappedValue = Linear(allHeadSize, config.hidden_size)
  }

  func transposeForScores(_ x: MLXArray) -> MLXArray {
    let batchSize = x.dim(0)
    let seqLength = x.dim(1)

    let newShape = [batchSize, seqLength, numAttentionHeads, attentionHeadSize]

    let reshaped = reshaped(x, newShape)
    return reshaped.transposed(0, 2, 1, 3)
  }

  func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let mixedQueryLayer = query(hiddenStates)
    let mixedKeyLayer = key(hiddenStates)
    let mixedValueLayer = value(hiddenStates)

    let queryLayer = transposeForScores(mixedQueryLayer)
    let keyLayer = transposeForScores(mixedKeyLayer)
    let valueLayer = transposeForScores(mixedValueLayer)

    var attentionScores = matmul(queryLayer, keyLayer.transposed(0, 1, 3, 2))
    attentionScores = attentionScores / MLXArray(sqrt(Float(attentionHeadSize)))

    if let mask = mask {
      attentionScores = attentionScores + mask
    }

    let attentionProbs = softmax(attentionScores, axis: -1)

    var contextLayer = matmul(attentionProbs, valueLayer)

    contextLayer = contextLayer.transposed(0, 2, 1, 3)

    let outputShape = [contextLayer.dim(0), contextLayer.dim(1), allHeadSize]
    return out_lin(contextLayer.reshaped(outputShape))
  }
}

private class FFN: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(_ config: DistilBertConfiguration) {
        _lin1.wrappedValue = Linear(config.hidden_size, config.intermediate_size)
        _lin2.wrappedValue = Linear(config.intermediate_size, config.hidden_size)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = lin1(x)
        x = gelu(x) 
        x = lin2(x)
        return x
    }
}

private class TransformerBlock: Module {
  @ModuleInfo(key: "attention") var attention: BertAttention
  @ModuleInfo(key: "sa_layer_norm") var sa_layer_norm: LayerNorm
  @ModuleInfo(key: "ffn") var ffn: FFN
  @ModuleInfo(key: "output_layer_norm") var output_layer_norm: LayerNorm

  init(_ config: DistilBertConfiguration) {
    _attention.wrappedValue = BertAttention(config)
    _sa_layer_norm.wrappedValue = LayerNorm(dimensions: config.hidden_size, eps: config.layer_norm_eps)
    _ffn.wrappedValue = FFN(config)
    _output_layer_norm.wrappedValue = LayerNorm(dimensions: config.hidden_size, eps: config.layer_norm_eps)
  }

  func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let attentionOut = attention(inputs, mask: mask)
    let saNormOut = sa_layer_norm(inputs + attentionOut) 
        let ffnOut = ffn(saNormOut)
     let outputNormOut = output_layer_norm(saNormOut + ffnOut)
    return outputNormOut
  }
}

private class BertEncoder: Module {
  @ModuleInfo(key: "layer") fileprivate var layer: [TransformerBlock]
  init(_ config: DistilBertConfiguration) {
    precondition(config.vocab_size > 0)
    _layer.wrappedValue = (0..<config.num_hidden_layers).map { _ in TransformerBlock(config) }
  }
  func callAsFunction(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
    var outputs = inputs
    for l in layer {
      outputs = l(outputs, mask: attentionMask)
    }
    return outputs
  }
}



public class DistilBertModel: Module, EmbeddingModel {
  @ModuleInfo(key: "embeddings") fileprivate var embedding: BertEmbedding
  fileprivate let transformer: BertEncoder

  public init(
    _ config: DistilBertConfiguration
  ) {
    transformer = BertEncoder(config)
    _embedding.wrappedValue = BertEmbedding(config)
  }

  fileprivate func getExtendedAttentionMask(_ attentionMask: MLXArray) -> MLXArray {
    if attentionMask.ndim == 3 {
      let extended = attentionMask.expandedDimensions(axes: [1])
      return (1.0 - extended) * -10000.0
    } else if attentionMask.ndim == 2 {
      let extended = attentionMask.expandedDimensions(axes: [1, 2])
      return (1.0 - extended) * -10000.0
    } else {
      fatalError("Wrong shape for attention_mask (shape \(attentionMask.shape))")
    }
  }

  public func callAsFunction(
    _ inputIds: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
    attentionMask: MLXArray? = nil
  )
    -> EmbeddingModelOutput
  {
    let (batchSize, seqLen) = (inputIds.dim(0), inputIds.dim(1))
    let embeddingOutput = embedding(inputIds: inputIds, tokenTypeIds: tokenTypeIds)
    let _mask = attentionMask ?? MLXArray.ones([batchSize, seqLen])

    let mask = getExtendedAttentionMask(_mask)

    let transformerOutput = transformer(
      embeddingOutput,
      attentionMask: mask)


    var text_embeds = meanPooling(
      lastHiddenState: transformerOutput,
      attentionMask: _mask)
    text_embeds = normalizeEmbeddings(text_embeds)

    return EmbeddingModelOutput(
      hiddenStates: transformerOutput,
      poolerOutput: nil,
      textEmbeds: text_embeds)
  }

  public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitizedWeights = [String: MLXArray]()
    for (key, value) in weights {
      if key.contains("position_ids") || key.contains("vocab_transform") || key.contains("vocab_layer_norm") || key.contains("vocab_projector") {
        continue
      } else {
        let sanitizedKey = key.replacingOccurrences(of: "distilbert.", with: "")
        sanitizedWeights[sanitizedKey] = value
      }
    }
    return sanitizedWeights
  }
}