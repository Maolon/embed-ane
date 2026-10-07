import AppKit
import EmbedANEAppSupport
import Observation

/// The Embed ANE chip glyph: a rounded package with a pin-1 mark around one die.
/// The die tells the state — solid when the model is in memory, outlined when it
/// loads on demand, absent when unloaded. Drawn as a template image so the menu
/// bar tints it; geometry is on an 18-point grid.
enum StatusGlyph {
    static func image(for kind: StatusKind, pulse: CGFloat? = nil, pointSize: CGFloat = 18) -> NSImage {
        let image = NSImage(size: NSSize(width: pointSize, height: pointSize), flipped: true) { _ in
            let scale = pointSize / 18
            NSGraphicsContext.current?.cgContext.scaleBy(x: scale, y: scale)
            draw(kind, pulse: pulse)
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "Embed ANE"
        return image
    }

    private static let package = NSRect(x: 1.8, y: 1.8, width: 14.4, height: 14.4)
    private static let die = NSRect(x: 5.6, y: 5.6, width: 6.8, height: 6.8)

    private static func draw(_ kind: StatusKind, pulse: CGFloat?) {
        let ink = NSColor.black.withAlphaComponent(kind == .stopped ? 0.45 : 1)
        ink.set()
        let outline = NSBezierPath(roundedRect: package, xRadius: 4.2, yRadius: 4.2)
        outline.lineWidth = 1.3
        if kind == .stopped { outline.setLineDash([2.4, 1.9], count: 2, phase: 0) }
        outline.stroke()
        NSBezierPath(ovalIn: NSRect(x: 3.9, y: 3.9, width: 1.7, height: 1.7)).fill()

        let core = NSBezierPath(roundedRect: die, xRadius: 1.7, yRadius: 1.7)
        core.lineWidth = 1.2
        switch kind {
        case .ready:
            core.fill()
        case .standby:
            core.stroke()
        case .working:
            core.stroke()
            if let pulse {
                NSColor.black.withAlphaComponent(pulse).set()
                core.fill()
            }
        case .error:
            let bar = NSBezierPath()
            bar.move(to: NSPoint(x: 9, y: 5.9)); bar.line(to: NSPoint(x: 9, y: 9.6))
            bar.lineWidth = 1.6; bar.lineCapStyle = .round
            bar.stroke()
            NSBezierPath(ovalIn: NSRect(x: 8, y: 11.1, width: 2, height: 2)).fill()
        case .unloaded, .stopped:
            break
        }
    }
}

/// Drives the die's fill while Working lasts longer than `WorkingAnimation.delay`.
/// Wakes on status changes; ticks only while it is animating.
@MainActor @Observable final class GlyphAnimator {
    /// Die fill opacity for the Working glyph; nil draws the outline only.
    private(set) var pulse: CGFloat?
    @ObservationIgnored private var task: Task<Void, Never>?

    func start(observing model: MenuBarModel) {
        guard task == nil else { return }
        task = Task { [weak self, weak model] in
            var since: ContinuousClock.Instant?
            var frame = 0
            while !Task.isCancelled {
                guard let self, let model else { return }
                let working = model.presentation.kind == .working
                since = working ? (since ?? .now) : nil
                guard working else {
                    self.pulse = nil; frame = 0
                    await Self.nextChange(of: model)
                    continue
                }
                if NSWorkspace.shared.accessibilityDisplayShouldReduceMotion {
                    self.pulse = 0.4
                } else if WorkingAnimation.shouldAnimate(workingSince: since, now: .now) {
                    // One breath every 1.2 s at 10 frames per second.
                    frame = (frame + 1) % 12
                    self.pulse = 0.1 + 0.75 * (0.5 - 0.5 * cos(2 * .pi * CGFloat(frame) / 12))
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    private static func nextChange(of model: MenuBarModel) async {
        await withCheckedContinuation { continuation in
            withObservationTracking { _ = model.presentation } onChange: { continuation.resume() }
        }
    }
}
