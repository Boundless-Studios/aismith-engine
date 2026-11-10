export class Mesh2MotionMapper {
     /**
   * Check if the given skeleton is a Mesh2Motion skeleton
   * All the source bones are Mesh2Motion, so we really just need 
   * to check if the target bones match the source bones
   * @param bones - Bones to check
   * @returns True if all the bones match
   */
  static is_source_bones_same_as_target (source_bone_names: string[], target_bone_names: string[]): boolean {
    return source_bone_names.every(name => target_bone_names.includes(name))
  }
}