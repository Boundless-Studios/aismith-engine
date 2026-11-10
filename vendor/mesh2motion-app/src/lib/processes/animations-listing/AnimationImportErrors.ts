// Custom error types for better error handling
export class NoAnimationsError extends Error {
  constructor (message: string = 'No animations found in the file.') {
    super(message)
    this.name = 'NoAnimationsError'
  }
}

export class IncompatibleSkeletonError extends Error {
  constructor (message: string = 'Animation skeleton is incompatible with target skeleton.') {
    super(message)
    this.name = 'IncompatibleSkeletonError'
  }
}

export class LoadError extends Error {
  constructor (message: string = 'Failed to load animation file.') {
    super(message)
    this.name = 'LoadError'
  }
}
