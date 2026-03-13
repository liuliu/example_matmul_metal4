import SwiftUI

@main
struct ConvolutionBenchApp: App {
  @StateObject private var runModel = RunModel()

  var body: some Scene {
    WindowGroup {
      ContentView(runModel: runModel)
    }
  }
}
