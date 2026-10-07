import EmbedANECore
import Foundation

/// OpenAI's four text/token forms plus our explicitly nonstandard content parts.
enum EmbeddingInput: Decodable {
    case texts([String]), tokens([[Int]]), parts([EmbeddingContentPart])
    init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let text = try? c.decode(String.self) { self = .texts([text]) }
        else if let texts = try? c.decode([String].self) { self = .texts(texts) }
        else if let tokens = try? c.decode([Int].self) { self = .tokens([tokens]) }
        else if let tokens = try? c.decode([[Int]].self) { self = .tokens(tokens) }
        else { self = .parts(try c.decode([EmbeddingContentPart].self)) }
    }
    func validate(maxTexts: Int) throws {
        switch self {
        case let .texts(texts):
            try validateBatch(texts.count, maximum: maxTexts)
            if let index = texts.firstIndex(where: \.isEmpty) { throw EmbedANEError.emptyInput(index: index) }
        case let .tokens(tokens):
            try validateBatch(tokens.count, maximum: maxTexts)
            if let index = tokens.firstIndex(where: \.isEmpty) { throw EmbedANEError.emptyInput(index: index) }
        case let .parts(parts):
            guard !parts.isEmpty else { throw EmbedANEError.emptyInput(index: nil) }
            guard parts.count <= 2048 else { throw EmbedANEError.invalidRequest("input may contain at most 2048 content parts.", param: "input") }
            let joint = jointContent
            guard joint.images.count + joint.videos.count <= 1 else {
                throw EmbedANEError.invalidRequest("Only one image or video per request is supported.", param: "input")
            }
            guard !joint.text.isEmpty || !joint.images.isEmpty || !joint.videos.isEmpty else { throw EmbedANEError.emptyInput(index: nil) }
        }
    }
    var jointContent: (text: String, images: [String], videos: [String]) {
        guard case let .parts(parts) = self else { return ("", [], []) }
        var texts: [String] = [], images: [String] = [], videos: [String] = []
        for part in parts {
            switch part {
            case let .text(text): texts.append(text)
            case let .image(url): images.append(url)
            case let .video(url): videos.append(url)
            }
        }
        // Concatenate text in order without inserting unrequested separators.
        // The prompt places the single image or video before the combined text.
        return (texts.joined(), images, videos)
    }
    private func validateBatch(_ count: Int, maximum: Int) throws {
        guard count > 0 else { throw EmbedANEError.emptyInput(index: nil) }
        guard count <= min(maximum, 2048) else { throw EmbedANEError.batchTooLarge(count) }
    }
}

enum EmbeddingContentPart: Decodable {
    case text(String), image(String), video(String)
    enum CodingKeys: String, CodingKey { case type, text, imageURL = "image_url", videoURL = "video_url" }
    /// `{"url": "..."}`, shared by image_url and video_url.
    private struct ImageURL: Decodable {
        let url: String
        enum CodingKeys: String, CodingKey { case url }
        init(from decoder: any Swift.Decoder) throws {
            try StrictCoding.rejectUnknown(decoder, allowed: ["url"])
            url = try decoder.container(keyedBy: CodingKeys.self).decode(String.self, forKey: .url)
        }
    }
    init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(String.self, forKey: .type) {
        case "text":
            try StrictCoding.rejectUnknown(decoder, allowed: ["type", "text"])
            self = .text(try c.decode(String.self, forKey: .text))
        case "image_url":
            try StrictCoding.rejectUnknown(decoder, allowed: ["type", "image_url"])
            self = .image(try c.decode(ImageURL.self, forKey: .imageURL).url)
        case "video_url":
            try StrictCoding.rejectUnknown(decoder, allowed: ["type", "video_url"])
            self = .video(try c.decode(ImageURL.self, forKey: .videoURL).url)
        default:
            throw EmbedANEError.invalidRequest("Content parts must have type text, image_url or video_url.", param: "input")
        }
    }
}
