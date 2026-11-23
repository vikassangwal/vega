import { Routes, Route } from 'react-router-dom'
import { Canvas } from '@react-three/fiber'
import { HolographicHUD } from '@components/HUD/HolographicHUD'
import { ChatPanel } from '@components/Panels/ChatPanel'
import { TaskPanel } from '@components/Panels/TaskPanel'
import { VisionPanel } from '@components/Panels/VisionPanel'
import { SystemPanel } from '@components/Panels/SystemPanel'
import { NotificationPanel } from '@components/Panels/NotificationPanel'
import { VoiceButton } from '@components/Controls/VoiceButton'
import { CommandPalette } from '@components/Controls/CommandPalette'
import { SettingsModal } from '@components/Controls/SettingsModal'
import { useWebSocket } from '@hooks/useWebSocket'
import { useVoice } from '@hooks/useVoice'
import { useVegaStore } from '@stores/vegaStore'
import { SettingsProvider } from '@stores/settingsStore'

function App() {
  const { connect, disconnect, isConnected } = useWebSocket()
  const { startListening, stopListening, isListening } = useVoice()
  const { setHUDVisible, setPanelOpen } = useVegaStore()

  // Initialize WebSocket connection
  React.useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return (
    <div className="vega-bg grid-bg min-h-screen relative overflow-hidden">
      {/* Scanline overlay */}
      <div className="scanlines" aria-hidden="true" />

      {/* 3D HUD Canvas */}
      <div className="fixed inset-0 z-10 pointer-events-none">
        <Canvas
          camera={{ position: [0, 0, 5], fov: 60 }}
          gl={{ antialias: true, alpha: true, preserveDrawingBuffer: false }}
          style={{ width: '100%', height: '100%' }}
        >
          <color attach="background" args={['#0a0a0f', 0]} />
          <fog attach="fog" args={['#0a0a0f', 1, 100]} />
          <HolographicHUD />
        </Canvas>
      </div>

      {/* UI Overlay Panels */}
      <div className="fixed inset-0 z-20 pointer-events-auto p-4 md:p-6 lg:p-8">
        <div className="flex h-full w-full gap-4">
          {/* Left Sidebar */}
          <aside className="flex flex-col gap-4 w-full md:w-72 lg:w-80 shrink-0">
            <SystemPanel />
            <VisionPanel />
          </aside>

          {/* Main Chat Area */}
          <main className="flex-1 flex flex-col min-w-0">
            <ChatPanel />
          </main>

          {/* Right Sidebar */}
          <aside className="flex flex-col gap-4 w-full md:w-72 lg:w-80 shrink-0">
            <TaskPanel />
            <NotificationPanel />
          </aside>
        </div>

        {/* Bottom Controls */}
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-4 z-30">
          <VoiceButton />
          <CommandPalette />
        </div>
      </div>

      {/* Modals */}
      <SettingsModal />

      {/* Connection Status */}
      <div className="fixed top-4 right-4 z-40 flex items-center gap-2">
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg glass-panel transition-all duration-300`}>
          <span className={`status-dot ${isConnected ? 'status-online' : 'status-offline'}`} />
          <span className="text-xs font-mono text-vega-accent/80">
            {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default App