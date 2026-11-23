import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface Settings {
  // Voice
  wakeWord: string
  wakeWordSensitivity: number
  voiceProvider: 'edge' | 'elevenlabs' | 'piper' | 'coqui'
  voiceId: string
  voiceSpeed: number
  voicePitch: number
  voiceVolume: number
  language: string
  bargeInEnabled: boolean

  // Vision
  visionEnabled: boolean
  faceRecognitionEnabled: boolean
  objectDetectionEnabled: boolean
  gestureEnabled: boolean
  detectionConfidence: number
  screenReadingEnabled: boolean

  // System
  autoStart: boolean
  startMinimized: boolean
  sentryMode: boolean
  proactiveSuggestions: boolean
  dailyBriefingTime: string
  briefingEnabled: boolean

  // UI
  hudOpacity: number
  hudScale: number
  showFPS: boolean
  showDebugInfo: boolean
  panelAnimations: boolean
  reducedMotion: boolean
  compactMode: boolean

  // Privacy
  localOnlyMode: boolean
  dataRetentionDays: number
  shareAnalytics: boolean

  // Advanced
  ollamaHost: string
  ollamaModel: string
  customModels: string[]
  developerMode: boolean
}

interface SettingsState {
  settings: Settings
  availableVoices: Record<string, Array<{ id: string; name: string; language: string }>>
  availableModels: string[]
  isLoading: boolean
  error: string | null

  // Actions
  updateSettings: (partial: Partial<Settings>) => void
  resetSettings: () => void
  loadSettings: () => Promise<void>
  saveSettings: () => Promise<void>
  fetchVoices: () => Promise<void>
  fetchModels: () => Promise<void>
  setError: (error: string | null) => void
}

const defaultSettings: Settings = {
  // Voice
  wakeWord: 'vega',
  wakeWordSensitivity: 0.5,
  voiceProvider: 'edge',
  voiceId: 'en-US-AriaNeural',
  voiceSpeed: 1.0,
  voicePitch: 0,
  voiceVolume: 1.0,
  language: 'en-US',
  bargeInEnabled: true,

  // Vision
  visionEnabled: true,
  faceRecognitionEnabled: true,
  objectDetectionEnabled: true,
  gestureEnabled: true,
  detectionConfidence: 0.5,
  screenReadingEnabled: true,

  // System
  autoStart: false,
  startMinimized: false,
  sentryMode: false,
  proactiveSuggestions: true,
  dailyBriefingTime: '07:00',
  briefingEnabled: true,

  // UI
  hudOpacity: 0.9,
  hudScale: 1.0,
  showFPS: false,
  showDebugInfo: false,
  panelAnimations: true,
  reducedMotion: false,
  compactMode: false,

  // Privacy
  localOnlyMode: false,
  dataRetentionDays: 30,
  shareAnalytics: false,

  // Advanced
  ollamaHost: 'http://localhost:11434',
  ollamaModel: 'llama3.1:70b',
  customModels: [],
  developerMode: false,
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      settings: defaultSettings,
      availableVoices: {},
      availableModels: [],
      isLoading: false,
      error: null,

      updateSettings: (partial) => set((state) => ({
        settings: { ...state.settings, ...partial }
      })),

      resetSettings: () => set({ settings: defaultSettings }),

      loadSettings: async () => {
        set({ isLoading: true, error: null })
        try {
          // In production, load from backend/localStorage
          // const saved = localStorage.getItem('vega-settings')
          // if (saved) set({ settings: JSON.parse(saved) })
          set({ isLoading: false })
        } catch (error) {
          set({ error: 'Failed to load settings', isLoading: false })
        }
      },

      saveSettings: async () => {
        try {
          const { settings } = get()
          // In production, save to backend/localStorage
          // localStorage.setItem('vega-settings', JSON.stringify(settings))
        } catch (error) {
          set({ error: 'Failed to save settings' })
        }
      },

      fetchVoices: async () => {
        set({ isLoading: true })
        try {
          // In production, fetch from voice engine
          // const response = await fetch('/api/voices')
          // const voices = await response.json()
          // set({ availableVoices: voices, isLoading: false })
          set({
            availableVoices: {
              edge: [
                { id: 'en-US-AriaNeural', name: 'Aria', language: 'en-US', gender: 'female' },
                { id: 'en-US-GuyNeural', name: 'Guy', language: 'en-US', gender: 'male' },
                { id: 'en-GB-SoniaNeural', name: 'Sonia', language: 'en-GB', gender: 'female' },
                { id: 'en-GB-RyanNeural', name: 'Ryan', language: 'en-GB', gender: 'male' },
                { id: 'hi-IN-SwaraNeural', name: 'Swara', language: 'hi-IN', gender: 'female' },
                { id: 'ja-JP-NanamiNeural', name: 'Nanami', language: 'ja-JP', gender: 'female' },
              ],
              elevenlabs: [
                { id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel', language: 'en', gender: 'female' },
                { id: 'pNInz6obpgDQGcFmaJgB', name: 'Adam', language: 'en', gender: 'male' },
              ],
              piper: [
                { id: 'en_US-amy-medium', name: 'Amy', language: 'en-US', gender: 'female' },
                { id: 'en_US-ryan-medium', name: 'Ryan', language: 'en-US', gender: 'male' },
              ],
              coqui: [
                { id: 'xtts_v2', name: 'XTTS v2 (Cloning)', language: 'multi', gender: 'any' },
              ],
            },
            isLoading: false
          })
        } catch (error) {
          set({ error: 'Failed to fetch voices', isLoading: false })
        }
      },

      fetchModels: async () => {
        try {
          // In production, fetch from Ollama
          set({
            availableModels: [
              'llama3.1:70b',
              'llama3.1:8b',
              'deepseek-coder:33b',
              'codellama:34b',
              'mistral:7b',
              'mixtral:8x7b',
              'phi3:14b',
              'gemma2:27b',
            ],
            isLoading: false
          })
        } catch (error) {
          set({ error: 'Failed to fetch models', isLoading: false })
        }
      },

      setError: (error) => set({ error }),
    }),
    {
      name: 'vega-settings',
      version: 1,
      migrate: (persistedState: any, version: number) => {
        if (version === 0) {
          return { ...persistedState, settings: { ...defaultSettings, ...persistedState.settings } }
        }
        return persistedState
      },
    }
  )
)