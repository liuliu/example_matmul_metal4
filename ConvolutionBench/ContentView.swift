import SwiftUI

struct ContentView: View {
  @ObservedObject var runModel: RunModel

  var body: some View {
    NavigationStack {
      VStack(alignment: .leading, spacing: 16) {
        HStack(spacing: 12) {
          Button(runModel.isRunning ? "Running..." : "Run Again") {
            runModel.start(force: true)
          }
          .buttonStyle(.borderedProminent)
          .disabled(runModel.isRunning)

          if let reportPath = runModel.reportPath {
            Text(reportPath)
              .font(.caption.monospaced())
              .foregroundStyle(.secondary)
              .lineLimit(2)
          }
        }

        ScrollView {
          Text(runModel.logText)
            .font(.system(.caption, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .leading)
            .textSelection(.enabled)
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
      }
      .padding()
      .navigationTitle("Convolution Bench")
      .task {
        runModel.start(force: false)
      }
    }
  }
}
