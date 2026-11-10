import type { HTMLAttributes, Key, Ref } from 'react'
import type { ModelViewerElement } from '@google/model-viewer'

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      'model-viewer': HTMLAttributes<HTMLElement> & {
        key?: Key
        ref?: Ref<ModelViewerElement>
        src?: string
        alt?: string
        loading?: 'auto' | 'lazy' | 'eager'
        exposure?: string
        'camera-controls'?: boolean
        'touch-action'?: string
        'interaction-prompt'?: string
        'camera-orbit'?: string
        'field-of-view'?: string
        'min-camera-orbit'?: string
        'max-camera-orbit'?: string
        'environment-image'?: string
        'shadow-intensity'?: string
      }
    }
  }
}
