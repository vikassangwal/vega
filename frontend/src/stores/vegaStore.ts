import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

export interface AgentStatus {
  name: string
  status: 'idle' | 'running' | 'processing' | 'error' | 'offline'
  progress?: number
  currentTask?: string
  lastUpdate: number
}

export interface VisionDetection {
  id: string
  type: 'object' | 'face' | 'gesture' | 'text' | 'screen'
  label: string
  confidence: number
  bbox: { x: number; y: number; width: number; height: number }
  timestamp: number
  metadata?: Record<string, unknown>
}

export interface SystemMetrics {
  cpu: number
  memory: number
  disk: number
  network: { up: number; down: number }
  gpu?: { usage: number; memory: number; temp: number }
  battery?: { level: number; charging: boolean }
  timestamp: number
}

export interface Notification {
  id: string
  type: 'info' | 'warning' | 'error' | 'success'
  title: string
  message: string
  timestamp: number
  read: boolean
  action?: { label: string; handler: () => void }
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  streaming?: boolean
  metadata?: Record<string, unknown>
}

export interface Task {
  id: string
  title: string
  description?: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  priority: 'low' | 'medium' | 'high' | 'urgent'
  agent?: string
  createdAt: number
  updatedAt: number
  progress?: number
}

interface VEGAState {
  // Connection
  isConnected: boolean
  connectionQuality: 'excellent' | 'good' | 'poor' | 'disconnected'

  // HUD State
  hudVisible: boolean
  hudMode: 'full' | 'compact' | 'minimal'

  // Agents
  agents: Record<string, AgentStatus>
  activeAgent: string | null

  // Vision
  detections: VisionDetection[]
  selectedDetection: VisionDetection | null

  // System
  systemMetrics: SystemMetrics | null

  // Chat
  messages: ChatMessage[]
  isStreaming: boolean

  // Tasks
  tasks: Task[]
  selectedTask: Task | null

  // Notifications
  notifications: Notification[]
  unreadCount: number

  // Voice
  isListening: boolean
  isSpeaking: boolean
  voiceVolume: number

  // Panels
  panelOpen: Record<string, boolean>

  // Settings
  theme: 'dark' | 'light'
  animationsEnabled: boolean
  reducedMotion: boolean

  // Actions
  setConnected: (connected: boolean) => void
  setConnectionQuality: (quality: VEGAState['connectionQuality']) => void
  setHUDVisible: (visible: boolean) => void
  setHUDMode: (mode: VEGAState['hudMode']) => void

  // Agent actions
  updateAgentStatus: (name: string, status: Partial<AgentStatus>) => void
  setActiveAgent: (name: string | null) => void

  // Vision actions
  addDetection: (detection: VisionDetection) => void
  clearDetections: () => void
  setSelectedDetection: (detection: VisionDetection | null) => void

  // System actions
  updateSystemMetrics: (metrics: SystemMetrics) => void

  // Chat actions
  addMessage: (message: ChatMessage) => void
  updateMessage: (id: string, content: string) => void
  setStreaming: (streaming: boolean) => void
  clearMessages: () => void

  // Task actions
  addTask: (task: Task) => void
  updateTask: (id: string, updates: Partial<Task>) => void
  removeTask: (id: string) => void
  setSelectedTask: (task: Task | null) => void

  // Notification actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void
  markNotificationRead: (id: string) => void
  clearNotifications: () => void

  // Voice actions
  setListening: (listening: boolean) => void
  setSpeaking: (speaking: boolean) => void
  setVoiceVolume: (volume: number) => void

  // Panel actions
  togglePanel: (panel: string) => void
  setPanelOpen: (panel: string, open: boolean) => void
  closeAllPanels: () => void

  // Settings
  setTheme: (theme: 'dark' | 'light') => void
  setAnimationsEnabled: (enabled: boolean) => void
  setReducedMotion: (enabled: boolean) => void

  // Reset
  reset: () => void
}

const initialState: Omit<VEGAState, 'setConnected' | 'setConnectionQuality' | 'setHUDVisible' | 'setHUDMode' | 'updateAgentStatus' | 'setActiveAgent' | 'addDetection' | 'clearDetections' | 'setSelectedDetection' | 'updateSystemMetrics' | 'addMessage' | 'updateMessage' | 'setStreaming' | 'clearMessages' | 'addTask' | 'updateTask' | 'removeTask' | 'setSelectedTask' | 'addNotification' | 'markNotificationRead' | 'clearNotifications' | 'setListening' | 'setSpeaking' | 'setVoiceVolume' | 'togglePanel' | 'setPanelOpen' | 'closeAllPanels' | 'setTheme' | 'setAnimationsEnabled' | 'setReducedMotion' | 'reset'> = {
  isConnected: false,
  connectionQuality: 'disconnected',
  hudVisible: true,
  hudMode: 'full',
  agents: {
    orchestrator: { name: 'orchestrator', status: 'offline', lastUpdate: 0 },
    executor: { name: 'executor', status: 'offline', lastUpdate: 0 },
    researcher: { name: 'researcher', status: 'offline', lastUpdate: 0 },
    creative: { name: 'creative', status: 'offline', lastUpdate: 0 },
    guardian: { name: 'guardian', status: 'offline', lastUpdate: 0 },
    planner: { name: 'planner', status: 'offline', lastUpdate: 0 },
    toolsmith: { name: 'toolsmith', status: 'offline', lastUpdate: 0 },
  },
  activeAgent: null,
  detections: [],
  selectedDetection: null,
  systemMetrics: null,
  messages: [],
  isStreaming: false,
  tasks: [],
  selectedTask: null,
  notifications: [],
  unreadCount: 0,
  isListening: false,
  isSpeaking: false,
  voiceVolume: 1,
  panelOpen: {
    chat: true,
    tasks: true,
    vision: true,
    system: true,
    notifications: true,
    settings: false,
  },
  theme: 'dark',
  animationsEnabled: true,
  reducedMotion: false,
}

export const useVegaStore = create<VEGAState>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    // Connection
    setConnected: (connected) => set({
      isConnected: connected,
      connectionQuality: connected ? 'excellent' : 'disconnected'
    }),

    setConnectionQuality: (quality) => set({ connectionQuality: quality }),

    // HUD
    setHUDVisible: (visible) => set({ hudVisible: visible }),
    setHUDMode: (mode) => set({ hudMode: mode }),

    // Agents
    updateAgentStatus: (name, status) => set((state) => ({
      agents: {
        ...state.agents,
        [name]: { ...state.agents[name], ...status, lastUpdate: Date.now() }
      }
    })),

    setActiveAgent: (name) => set({ activeAgent: name }),

    // Vision
    addDetection: (detection) => set((state) => ({
      detections: [...state.detections.slice(-99), detection]
    })),

    clearDetections: () => set({ detections: [] }),

    setSelectedDetection: (detection) => set({ selectedDetection: detection }),

    // System
    updateSystemMetrics: (metrics) => set({ systemMetrics: metrics }),

    // Chat
    addMessage: (message) => set((state) => ({
      messages: [...state.messages, message]
    })),

    updateMessage: (id, content) => set((state) => ({
      messages: state.messages.map(m => m.id === id ? { ...m, content } : m)
    })),

    setStreaming: (streaming) => set({ isStreaming: streaming }),

    clearMessages: () => set({ messages: [] }),

    // Tasks
    addTask: (task) => set((state) => ({
      tasks: [task, ...state.tasks]
    })),

    updateTask: (id, updates) => set((state) => ({
      tasks: state.tasks.map(t => t.id === id ? { ...t, ...updates, updatedAt: Date.now() } : t),
      selectedTask: state.selectedTask?.id === id ? { ...state.selectedTask, ...updates } : state.selectedTask
    })),

    removeTask: (id) => set((state) => ({
      tasks: state.tasks.filter(t => t.id !== id),
      selectedTask: state.selectedTask?.id === id ? null : state.selectedTask
    })),

    setSelectedTask: (task) => set({ selectedTask: task }),

    // Notifications
    addNotification: (notification) => set((state) => ({
      notifications: [
        { ...notification, id: crypto.randomUUID(), timestamp: Date.now(), read: false },
        ...state.notifications.slice(0, 99)
      ],
      unreadCount: state.unreadCount + 1
    })),

    markNotificationRead: (id) => set((state) => ({
      notifications: state.notifications.map(n => n.id === id ? { ...n, read: true } : n),
      unreadCount: Math.max(0, state.unreadCount - 1)
    })),

    clearNotifications: () => set({ notifications: [], unreadCount: 0 }),

    // Voice
    setListening: (listening) => set({ isListening: listening }),
    setSpeaking: (speaking) => set({ isSpeaking: speaking }),
    setVoiceVolume: (volume) => set({ voiceVolume: Math.max(0, Math.min(1, volume)) }),

    // Panels
    togglePanel: (panel) => set((state) => ({
      panelOpen: { ...state.panelOpen, [panel]: !state.panelOpen[panel] }
    })),

    setPanelOpen: (panel, open) => set((state) => ({
      panelOpen: { ...state.panelOpen, [panel]: open }
    })),

    closeAllPanels: () => set((state) => {
      const closed = { ...state.panelOpen }
      Object.keys(closed).forEach(k => closed[k] = false)
      return { panelOpen: closed }
    }),

    // Settings
    setTheme: (theme) => set({ theme }),
    setAnimationsEnabled: (enabled) => set({ animationsEnabled: enabled }),
    setReducedMotion: (enabled) => set({ reducedMotion: enabled }),

    // Reset
    reset: () => set(initialState),
  }))
)

// Selectors for performance
export const useConnected = () => useVegaStore(s => s.isConnected)
export const useAgents = () => useVegaStore(s => s.agents)
export const useDetections = () => useVegaStore(s => s.detections)
export const useSystemMetrics = () => useVegaStore(s => s.systemMetrics)
export const useMessages = () => useVegaStore(s => s.messages)
export const useTasks = () => useVegaStore(s => s.tasks)
export const useNotifications = () => useVegaStore(s => s.notifications)
export const useUnreadCount = () => useVegaStore(s => s.unreadCount)
export const useVoiceState = () => useVegaStore(s => ({
  isListening: s.isListening,
  isSpeaking: s.isSpeaking,
  volume: s.voiceVolume
}))
export const usePanelState = () => useVegaStore(s => s.panelOpen)
export const useHUDState = () => useVegaStore(s => ({
  visible: s.hudVisible,
  mode: s.hudMode
}))