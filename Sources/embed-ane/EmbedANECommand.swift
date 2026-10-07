import Darwin
import EmbedANECore
import Foundation

@main
struct EmbedANECommand {
    static func main() async {
        let environment = ProcessInfo.processInfo.environment
        let home: URL
        if let path = environment["HOME"], path.hasPrefix("/") { home = URL(fileURLWithPath: path, isDirectory: true) }
        else { home = FileManager.default.homeDirectoryForCurrentUser }
        let result = await CLIApplication.run(arguments: Array(CommandLine.arguments.dropFirst()),
            store: ConfigurationStore(home: home),
            workingDirectory: URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true),
            environment: environment, executor: ProductionCommands())
        do {
            if !result.stdout.isEmpty { try FileHandle.standardOutput.write(contentsOf: result.stdout) }
            if !result.stderr.isEmpty { try FileHandle.standardError.write(contentsOf: result.stderr) }
        } catch { Darwin.exit(1) }
        Darwin.exit(result.exitCode)
    }
}
