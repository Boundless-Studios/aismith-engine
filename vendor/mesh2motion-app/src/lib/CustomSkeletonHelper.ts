// Original code from
// https://github.com/mrdoob/three.js/blob/master/src/helpers/SkeletonHelper.js
// and ideas from
// https://discourse.threejs.org/t/extend-skeletonhelper-to-accommodate-fat-lines-perhaps-with-linesegments2/59436/2

import { Color, Matrix4, Vector3, Points, PointsMaterial, BufferGeometry, Float32BufferAttribute, TextureLoader, LineSegments, LineBasicMaterial, type ColorRepresentation } from 'three'
import { BoneCategory, BoneClassifier } from './solvers/BoneClassifier'
import { Utility } from './Utilities'
import { classifyBoneDisplay } from './BoneDisplayPolicy'

const _vector = /*@__PURE__*/ new Vector3()
const _boneMatrix = /*@__PURE__*/ new Matrix4()
const _matrixWorldInv = /*@__PURE__*/ new Matrix4()

interface CustomSkeletonHelperOptions {
  color?: ColorRepresentation
  jointColor?: ColorRepresentation
}

const bone_category_colors: Record<BoneCategory, number> = {
  [BoneCategory.Torso]: 0xff9f1c,
  [BoneCategory.Limb]: 0x3a86ff,
  [BoneCategory.Extremity]: 0x8338ec,
  [BoneCategory.Other]: 0xadb5bd
}

const display_role_colors = {
  axial: 0xff9f1c,
  clavicle: 0x22c55e,
  helper: 0x6b7280
} as const

class CustomSkeletonHelper extends LineSegments {
  private readonly joint_points: Points
  private readonly helper_points: Points
  private readonly joint_bone_indices: number[]
  private readonly helper_bone_indices: number[]
  private readonly jointTexture = new TextureLoader().load('/images/skeleton-joint-point.png')

  constructor (object: any, options: CustomSkeletonHelperOptions = {}) {
    const bones = getBoneList(object)
    const geometry = new BufferGeometry()

    const vertices = []
    const colors = []
    const color = new Color(options.color || 0x0000ff) // Default color blue
    const joint_color = new Color(options.jointColor ?? 0x0000ff) // Default joint color blue

    for (let i = 0; i < bones.length; i++) {
      const bone = bones[i]

      if (bone.parent && bone.parent.isBone) {
        vertices.push(0, 0, 0) // Start
        vertices.push(0, 0, 0) // End
        colors.push(color.r, color.g, color.b)
        colors.push(color.r, color.g, color.b)
      }
    }

    geometry.setAttribute('position', new Float32BufferAttribute(vertices, 3))
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3))

    const material = new LineBasicMaterial({
      vertexColors: true,
      depthTest: false,
      depthWrite: false,
      transparent: true
    })

    super(geometry, material)

    this.isSkeletonHelper = true
    this.type = 'CustomSkeletonHelper'

    this.root = object
    this.bones = bones

    this.matrix = object.matrixWorld
    this.matrixAutoUpdate = false

    // Add points for joints
    const make_points_material = (size: number): PointsMaterial => new PointsMaterial({
      size,
      color: options.jointColor !== undefined ? joint_color : 0xffffff,
      depthTest: false,
      sizeAttenuation: false, // Disable size attenuation to keep size constant in screen space
      map: this.jointTexture,
      transparent: true, // Enable transparency for the circular texture
      opacity: 0.8,
      vertexColors: options.jointColor === undefined
    })

    this.joint_bone_indices = []
    this.helper_bone_indices = []
    bones.forEach((bone, idx) => {
      const indices = classifyBoneDisplay(bone.name) === 'helper'
        ? this.helper_bone_indices
        : this.joint_bone_indices
      indices.push(idx)
    })

    const pointsGeometry = new BufferGeometry()
    pointsGeometry.setAttribute('position', new Float32BufferAttribute(this.joint_bone_indices.length * 3, 3))
    const helperGeometry = new BufferGeometry()
    helperGeometry.setAttribute('position', new Float32BufferAttribute(this.helper_bone_indices.length * 3, 3))

    // use bone category to color joints to help with seeing a bunch of them
    if (options.jointColor === undefined) {
      const bone_classifier = new BoneClassifier(bones as Bone[])
      const point_colors: number[] = []
      const helper_colors: number[] = []

      bones.forEach((bone, idx) => {
        const display_role = classifyBoneDisplay(bone.name)
        const role_color = display_role === 'limb'
          ? bone_category_colors[bone_classifier.get_category(idx)]
          : display_role_colors[display_role]
        const category_color = new Color(role_color)
        const target = display_role === 'helper' ? helper_colors : point_colors
        target.push(category_color.r, category_color.g, category_color.b)
      })

      pointsGeometry.setAttribute('color', new Float32BufferAttribute(point_colors, 3))
      helperGeometry.setAttribute('color', new Float32BufferAttribute(helper_colors, 3))
    }

    this.joint_points = new Points(pointsGeometry, make_points_material(30))
    this.helper_points = new Points(helperGeometry, make_points_material(12))
    this.helper_points.visible = false
    this.add(this.joint_points)
    this.add(this.helper_points)
  }

  updateMatrixWorld (force: boolean): void {
    const bones = this.bones
    const pointPositions = this.joint_points.geometry.getAttribute('position')
    const helperPositions = this.helper_points.geometry.getAttribute('position')

    const geometry = this.geometry
    const positions = geometry.getAttribute('position')

    _matrixWorldInv.copy(this.root.matrixWorld).invert()

    let lineIndex = 0
    for (let i = 0; i < bones.length; i++) {
      const bone = bones[i]
      _boneMatrix.multiplyMatrices(_matrixWorldInv, bone.matrixWorld)
      _vector.setFromMatrixPosition(_boneMatrix)

      const joint_index = this.joint_bone_indices.indexOf(i)
      const helper_index = this.helper_bone_indices.indexOf(i)
      if (joint_index >= 0) pointPositions.setXYZ(joint_index, _vector.x, _vector.y, _vector.z)
      if (helper_index >= 0) helperPositions.setXYZ(helper_index, _vector.x, _vector.y, _vector.z)

      if (bone.parent && bone.parent.isBone) {
        _boneMatrix.multiplyMatrices(_matrixWorldInv, bone.parent.matrixWorld)
        _vector.setFromMatrixPosition(_boneMatrix)
        positions.setXYZ(lineIndex * 2, _vector.x, _vector.y, _vector.z)

        _boneMatrix.multiplyMatrices(_matrixWorldInv, bone.matrixWorld)
        _vector.setFromMatrixPosition(_boneMatrix)
        positions.setXYZ(lineIndex * 2 + 1, _vector.x, _vector.y, _vector.z)
        lineIndex++
      }
    }

    pointPositions.needsUpdate = true
    helperPositions.needsUpdate = true
    positions.needsUpdate = true

    // Update bounding box and bounding sphere
    // otherwise the skeleton will be hidden when root bone on ground is off camera
    geometry.computeBoundingBox()
    geometry.computeBoundingSphere()

    super.updateMatrixWorld(force)
  }

  dispose (): void {
    this.geometry.dispose()
    this.material.dispose()
    this.joint_points.geometry.dispose()
    this.joint_points.material.dispose()
    this.helper_points.geometry.dispose()
    this.helper_points.material.dispose()
  }

  public show(): void {
    this.visible = true
  }

  public hide(): void {
    this.visible = false
  }

  public setJointsVisible (visible: boolean): void {
    this.joint_points.visible = visible
  }

  public setHelpersVisible (visible: boolean): void {
    this.helper_points.visible = visible
  }

  public setHideRightSideJoints (_value: boolean): void {
    // Mirror mode still shows both sides so the visible guide matches the
    // complete skeleton. Editing either side continues to mirror the change.
  }
}

function getBoneList (object: any): any[] {
  const boneList: any[] = []

  if (object.isBone === true) {
    boneList.push(object)
  }

  for (let i = 0; i < object.children.length; i++) {
    boneList.push.apply(boneList, getBoneList(object.children[i]))
  }

  return boneList
}

export { CustomSkeletonHelper }
