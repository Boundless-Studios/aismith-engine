export type BoneDisplayRole = 'helper' | 'axial' | 'clavicle' | 'limb'

export function classifyBoneDisplay (name: string): BoneDisplayRole {
  const normalized = name.trim().toLowerCase()
  if (/(?:^|[_.-])(?:leaf|end|tip)(?:$|[_.-])/.test(normalized) || normalized.endsWith('end')) {
    return 'helper'
  }
  if (normalized.includes('clavicle') || normalized.includes('collar')) {
    return 'clavicle'
  }
  if (['spine', 'pelvis', 'hips', 'neck', 'head', 'chest', 'torso'].some(part => normalized.includes(part))) {
    return 'axial'
  }
  return 'limb'
}
