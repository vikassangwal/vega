import { useEffect, useRef, useCallback, useState } from 'react'
import { io, Socket } from 'socket.io-client'
import { useVegaStore } from '@stores/vegaStore'
import { useSettingsStore } from '@stores/settingsStore'

interface WebSocketMessage {
  type: string
  payload: unknown
  requestId?: string
  timestamp: number
}

interface UseWebSocketReturn {
  isConnected: boolean
  connectionQuality: 'excellent' | 'good' | 'poor' | 'disconnected'
  send: (type: string, payload: unknown, requestId?: string) => Promise<unknown>
  on: (event: string, handler: (data: unknown) => void) => () => void
  emit: (event: string, data: unknown) => void
}

export function useWebSocket(): UseWebSocketReturn {
  const socketRef = useRef<Socket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 10
  const reconnectDelayRef = useRef(1000)
  const handlersRef = useRef<Map<string, Set<(data: unknown) => void>>>(new Map())
  const pendingRequestsRef = useRef<Map<string, { resolve: (value: unknown) => void; reject: (error: Error) => void; timeout: NodeJS.Timeout }>>(new Map())
  const [isConnected, setIsConnected] = useState(false)
  const [connectionQuality, setConnectionQuality] = useState<'excellent' | 'good' | 'poor' | 'disconnected'>('disconnected')
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastPingRef = useRef<number>(0)

  const { setConnected, setConnectionQuality, updateAgentStatus, addDetection, updateSystemMetrics, addMessage, setStreaming, addNotification, updateTask } = useVegaStore()
  const { settings } = useSettingsStore()

  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = settings.ollamaHost.replace('http://', '').replace('https://', '') || 'localhost:8000'
    return `${protocol}//${host}/ws`
  }, [settings.ollamaHost])

  const emit = useCallback((event: string, data: unknown) => {
    socketRef.current?.emit(event, data)
  }, [])

  const on = useCallback((event: string, handler: (data: unknown) => void) => {
    if (!handlersRef.current.has(event)) {
      handlersRef.current.set(event, new Set())
    }
    handlersRef.current.get(event)!.add(handler)

    return () => {
      handlersRef.current.get(event)?.delete(handler)
    }
  }, [])

  const emitToServer = useCallback((type: string, payload: unknown, requestId?: string): Promise<unknown> => {
    return new Promise((resolve, reject) => {
      if (!socketRef.current?.connected) {
        reject(new Error('WebSocket not connected'))
        return
      }

      const id = requestId || crypto.randomUUID()
      const timeout = setTimeout(() => {
        pendingRequestsRef.current.delete(id)
        reject(new Error('Request timeout'))
      }, 30000)

      pendingRequestsRef.current.set(id, { resolve, reject, timeout })

      socketRef.current.emit('message', { type, payload, requestId: id, timestamp: Date.now() })
    })
  }, [])

  const send = useCallback(async (type: string, payload: unknown, requestId?: string) => {
    return emitToServer(type, payload, requestId)
  }, [emitToServer])

  const handleMessage = useCallback((message: WebSocketMessage) => {
    const { type, payload, requestId } = message

    // Handle request responses
    if (requestId && pendingRequestsRef.current.has(requestId)) {
      const { resolve, reject, timeout } = pendingRequestsRef.current.get(requestId)!
      clearTimeout(timeout)
      pendingRequestsRef.current.delete(requestId)
      resolve(payload)
      return
    }

    // Handle event handlers
    const handlers = handlersRef.current.get(type)
    if (handlers) {
      handlers.forEach(handler => handler(payload))
    }

    // Handle built-in message types
    switch (type) {
      case 'agent_status':
        updateAgentStatus(payload as { name: string; status: any })
        break
      case 'vision_detection':
        addDetection(payload as any)
        break
      case 'system_metrics':
        updateSystemMetrics(payload as any)
        break
      case 'chat_message':
        addMessage(payload as any)
        break
      case 'chat_stream_chunk':
        // Handle streaming
        break
      case 'notification':
        addNotification(payload as any)
        break
      case 'task_update':
        updateTask(payload as any)
        break
      case 'briefing':
        // Handle briefing
        break
      case 'anomaly':
        // Handle anomaly
        break
      case 'suggestion':
        // Handle suggestion
        break
      case 'voice_state':
        // Handle voice state
        break
      case 'pong':
        // Calculate latency
        const latency = Date.now() - lastPingRef.current
        setConnectionQuality(latency < 100 ? 'excellent' : latency < 300 ? 'good' : 'poor')
        break
    }
  }, [updateAgentStatus, addDetection, updateSystemMetrics, addMessage, addNotification, updateTask])

  const connect = useCallback(() => {
    if (socketRef.current?.connected) return

    const url = getWebSocketUrl()
    console.log('[WS] Connecting to:', url)

    socketRef.current = io(url, {
      transports: ['websocket', 'polling'],
      reconnection: false, // We handle reconnection manually
      timeout: 10000,
      auth: {
        clientVersion: '1.0.0',
        clientId: localStorage.getItem('vega-client-id') || crypto.randomUUID(),
      },
    })

    socketRef.current.on('connect', () => {
      console.log('[WS] Connected')
      setIsConnected(true)
      setConnectionQuality('excellent')
      reconnectAttemptsRef.current = 0
      reconnectDelayRef.current = 1000

      // Start ping interval
      pingIntervalRef.current = setInterval(() => {
        if (socketRef.current?.connected) {
          lastPingRef.current = Date.now()
          socketRef.current.emit('ping', { timestamp: lastPingRef.current })
        }
      }, 30000)

      // Request initial state
      socketRef.current.emit('message', {
        type: 'get_state',
        payload: {},
        timestamp: Date.now(),
      })
    }, [getWebSocketUrl])

    socketRef.current.on('disconnect', (reason) => {
      console.log('[WS] Disconnected:', reason)
      setIsConnected(false)
      setConnectionQuality('disconnected')

      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current)
        pingIntervalRef.current = null
      }

      // Attempt reconnection
      if (reconnectAttemptsRef.current < 10) {
        const delay = reconnectDelayRef.current
        reconnectAttemptsRef.current++
        reconnectDelayRef.current = Math.min(delay * 1.5, 30000)

        setTimeout(() => {
          connect()
        }, delay)
      } else {
        setConnectionQuality('disconnected')
      }
    })

    socketRef.current.on('connect_error', (error) => {
      console.error('[WS] Connection error:', error)
      setConnectionQuality('poor')
    })

    socketRef.current.on('message', handleMessage)
  }, [getWebSocketUrl, handleMessage])

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
    setIsConnected(false)
    setConnectionQuality('disconnected')
  }, [])

  // Initialize connection
  useEffect(() => {
    connect()

    return () => {
      disconnect()
      // Clean up pending requests
      pendingRequestsRef.current.forEach(({ reject, timeout }) => {
        clearTimeout(timeout)
        reject(new Error('WebSocket disconnected'))
      })
      pendingRequestsRef.current.clear()
      handlersRef.current.clear()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    connectionQuality,
    send,
    on,
    emit,
  }
}