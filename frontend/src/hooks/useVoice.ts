import { useEffect, useRef, useCallback, useState } from 'react'
import { useVegaStore } from '@stores/vegaStore'
import { useSettingsStore } from '@stores/settingsStore'

interface UseVoiceReturn {
  isListening: boolean
  isSpeaking: boolean
  volume: number
  startListening: () => Promise<void>
  stopListening: () => void
  speak: (text: string, options?: { emotion?: string; priority?: number }) => Promise<void>
  cancelSpeech: () => void
  setVolume: (volume: number) => void
}

export function useVoice(): UseVoiceReturn {
  const { setListening, setSpeaking, setVoiceVolume, isListening, isSpeaking, voiceVolume } = useVegaStore()
  const { settings } = useSettingsStore()

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null)
  const silenceThresholdRef = useRef(0.01)
  const silenceDurationRef = useRef(1500) // ms
  const [isSupported, setIsSupported] = useState(false)
  const [permission, setPermission] = useState<'granted' | 'denied' | 'prompt'>('prompt')

  // Check browser support
  useEffect(() => {
    const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
    setIsSupported(supported)

    if (supported) {
      navigator.permissions.query({ name: 'microphone' as PermissionName }).then(result => {
        setPermission(result.state)
        result.onchange = () => setPermission(result.state)
      })
    }
  }, [])

  // Initialize audio context
  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
    }
    return audioContextRef.current
  }, [])

  // Start listening for wake word / commands
  const startListening = useCallback(async () => {
    if (isListening || !isSupported) return

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1,
        },
      })

      const audioContext = initAudioContext()
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)

      // Silence detection
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0)
        let sum = 0
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i]
        }
        const rms = Math.sqrt(sum / inputData.length)

        if (rms > silenceThresholdRef.current) {
          // Voice detected
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current)
            silenceTimerRef.current = null
          }
        } else if (!silenceTimerRef.current) {
          // Start silence timer
          silenceTimerRef.current = setTimeout(() => {
            // Silence detected, could trigger auto-stop
            console.log('[Voice] Silence detected, stopping...')
            stopListening()
          }, silenceDurationRef.current)
        }
      }

      source.connect(processor)
      processor.connect(audioContext.destination)

      // Also record for STT
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' })
      mediaRecorderRef.current = recorder
      audioChunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data)
        }
      }

      recorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        // Send to STT engine via WebSocket
        // This would integrate with the backend STT
      }

      recorder.start(100) // Collect chunks every 100ms
      setListening(true)

    } catch (error) {
      console.error('[Voice] Failed to start listening:', error)
    }
  }, [isListening, isSupported, setListening])

  const stopListening = useCallback(() => {
    if (!isListening) return

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current)
      silenceTimerRef.current = null
    }

    // Clean up audio context
    if (mediaRecorderRef.current?.stream) {
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())
    }

    setListening(false)
  }, [isListening, setListening])

  // Speak text using TTS
  const speak = useCallback(async (text: string, options?: { emotion?: string; priority?: number }) => {
    if (!text.trim()) return

    setSpeaking(true)

    try {
      // Use the TTS engine via WebSocket
      // This would send to the backend TTS engine
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voice: settings.voiceId,
          speed: settings.voiceSpeed,
          pitch: settings.voicePitch,
          emotion: options?.emotion,
          format: 'mp3',
        }),
      })

      if (!response.ok) throw new Error('TTS request failed')

      const audioBlob = await response.blob()
      const audioUrl = URL.createObjectURL(audioBlob)
      const audio = new Audio(audioUrl)
      audio.volume = settings.voiceVolume

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl)
        setSpeaking(false)
      }

      audio.onerror = () => {
        URL.revokeObjectURL(audioUrl)
        setSpeaking(false)
      }

      await audio.play()

    } catch (error) {
      console.error('[Voice] Speech synthesis failed:', error)
      setSpeaking(false)

      // Fallback to browser SpeechSynthesis
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text)
        utterance.rate = settings.voiceSpeed
        utterance.pitch = 1 + settings.voicePitch / 10
        utterance.volume = settings.voiceVolume
        utterance.onend = () => setSpeaking(false)
        utterance.onerror = () => setSpeaking(false)
        speechSynthesis.speak(utterance)
      }
    }
  }, [settings, setSpeaking])

  const cancelSpeech = useCallback(() => {
    speechSynthesis.cancel()
    setSpeaking(false)
  }, [setSpeaking])

  const setVolume = useCallback((volume: number) => {
    setVoiceVolume(Math.max(0, Math.min(1, volume)))
  }, [setVoiceVolume])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening()
      cancelSpeech()
    }
  }, [stopListening, cancelSpeech])

  return {
    isListening,
    isSpeaking,
    volume: voiceVolume,
    startListening,
    stopListening,
    speak,
    cancelSpeech,
    setVolume,
  }
}