// Copied from https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/Embedders/Configuration.swift

import Foundation

public enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Int.self) {
            self = .int(v)
        } else if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else if let v = try? values.decode([Int].self) {
            self = .ints(v)
        } else if let v = try? values.decode([Float].self) {
            self = .floats(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        case .ints(let v): try container.encode(v)
        case .floats(let v): try container.encode(v)
        }
    }

    /// Return the value as an optional array of integers.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInts() -> [Int]? {
        switch self {
        case .string(let string): nil
        case .int(let v): [v]
        case .float(let float): nil
        case .ints(let array): array
        case .floats(let array): nil
        }
    }

    /// Return the value as an optional integer.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInt() -> Int? {
        switch self {
        case .string(let string): nil
        case .int(let v): v
        case .float(let float): nil
        case .ints(let array): array.count == 1 ? array[0] : nil
        case .floats(let array): nil
        }
    }

    /// Return the value as an optional array of floats.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloats() -> [Float]? {
        switch self {
        case .string(let string): nil
        case .int(let v): [Float(v)]
        case .float(let float): [float]
        case .ints(let array): array.map { Float($0) }
        case .floats(let array): array
        }
    }

    /// Return the value as an optional float.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloat() -> Float? {
        switch self {
        case .string(let string): nil
        case .int(let v): Float(v)
        case .float(let float): float
        case .ints(let array): array.count == 1 ? Float(array[0]) : nil
        case .floats(let array): array.count == 1 ? array[0] : nil
        }
    }
}

private class ModelTypeRegistry: @unchecked Sendable {

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    private var creators: [String: @Sendable (URL) throws -> EmbeddingModel] = [
        "bert": {
            url in
            let configuration = try JSONDecoder().decode(
                BertConfiguration.self, from: Data(contentsOf: url))
            let model = BertModel(configuration)
            return model
        },
        "roberta": {
            url in
            let configuration = try JSONDecoder().decode(
                BertConfiguration.self, from: Data(contentsOf: url))
            let model = BertModel(configuration)
            return model
        },
        "xlm-roberta": {
            url in
            let configuration = try JSONDecoder().decode(
                BertConfiguration.self, from: Data(contentsOf: url))
            let model = BertModel(configuration)
            return model
        },
        "distilbert": {
            url in
            let configuration = try JSONDecoder().decode(
                DistilBertConfiguration.self, from: Data(contentsOf: url))
            let model = DistilBertModel(configuration)
            return model
        },
        "qwen2": {
            url in
            let configuration = try JSONDecoder().decode(
                Qwen2Configuration.self, from: Data(contentsOf: url))
            let model = Qwen2Model(configuration)
            return model
        },
    ]

    public func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> EmbeddingModel
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    public func createModel(configuration: URL, rawValue: String) throws -> EmbeddingModel {
        let creator = lock.withLock {
            creators[rawValue]
        }
        guard let creator else {
            throw EmbedderError(message: "Unsupported model type.")
        }
        return try creator(configuration)
    }

}

private let modelTypeRegistry = ModelTypeRegistry()

public struct ModelType: RawRepresentable, Codable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public static func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> EmbeddingModel
    ) {
        modelTypeRegistry.registerModelType(type, creator: creator)
    }

    public func createModel(configuration: URL) throws -> EmbeddingModel {
        try modelTypeRegistry.createModel(configuration: configuration, rawValue: rawValue)
    }
}

public struct BaseConfiguration: Codable, Sendable {
    public let modelType: ModelType

    public struct Quantization: Codable, Sendable {
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        let groupSize: Int
        let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
        }
    }

    public var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
    }
}