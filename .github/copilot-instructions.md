# GitHub Copilot Instructions for MLX Embeddings Swift Library

## Project Overview
This is a Swift library built with MLX for creating and using embedding models. The library provides a modern Swift interface for generating, manipulating, and utilizing embeddings in machine learning applications.

## Code Style Guidelines

### Swift 6 Best Practices
- Use Swift 6's latest features including improved result builders, parameter packs, and macro capabilities
- Leverage Swift's strong type system with appropriate use of generics, protocols, and type constraints
- Utilize Swift concurrency with async/await, actors, and structured concurrency patterns
- Prefer value types (structs) over reference types (classes) when appropriate
- Use property wrappers and newer Swift features to reduce boilerplate

### Self-Explanatory Code
- Write code that explains itself without relying on comments
- Choose descriptive variable, function, and type names that reveal intent
- Create small, focused functions with clear purposes
- Use Swift's expressive syntax to make code read like natural language
- Break complex logic into well-named helper functions
- Prefer explicit over implicit behavior

### SOLID Principles
- **Single Responsibility**: Each class/type should have only one reason to change
- **Open/Closed**: Types should be open for extension but closed for modification
- **Liskov Substitution**: Subtypes must be substitutable for their base types
- **Interface Segregation**: Prefer multiple specific protocols over one general protocol
- **Dependency Inversion**: Depend on abstractions, not concretions

## MLX Embeddings-Specific Guidelines
- Optimize tensor operations for performance and memory efficiency
- Design clear interfaces for model loading, embedding generation, and similarity calculations
- Implement proper error handling for model loading and inference failures
- Ensure memory management is appropriate for handling large embedding models
- Create intuitive APIs for common embedding tasks (similarity search, clustering, etc.)
- Provide appropriate abstractions for different embedding model architectures
- Include sensible defaults while allowing customization of embedding parameters

## Example Patterns

When designing model interfaces:
```swift
protocol EmbeddingModel {
    func encode(text: String) async throws -> Embedding
    func batchEncode(texts: [String]) async throws -> [Embedding]
    var dimensions: Int { get }
}
```

When implementing embedding vector operations:
```swift
struct Embedding {
    let vector: MLXArray
    
    func similarity(to other: Embedding) -> Float {
        cosineDistance(self.vector, other.vector)
    }
    
    func normalize() -> Embedding {
        Embedding(vector: normalize(self.vector))
    }
}
```

When creating factory methods:
```swift
extension EmbeddingModel {
    static func load(from url: URL) async throws -> Self {
        // Implementation that loads model weights and configuration
    }
    
    static func defaultModel() async throws -> Self {
        // Implementation that returns a standard pre-trained model
    }
}
```
