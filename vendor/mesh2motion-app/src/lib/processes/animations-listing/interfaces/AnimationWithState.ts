import { type AnimationClip } from 'three'
import { type AnimationClipMetadata } from './TransformedAnimationClipPair'

export interface AnimationWithState extends AnimationClip {
  isChecked?: boolean
  name: string
  metadata: AnimationClipMetadata
}
