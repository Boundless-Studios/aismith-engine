import * as THREE from 'three'

export class WebMRecorder {
  public width: number
  public height: number
  public fps: number = 24
  public kbps: number = 2600

  public readonly recorded_files: File[] = []
  private readonly renderer_ref: THREE.WebGLRenderer

  private old_size: THREE.Vector2 | null = null
  private old_pr: number | null = null
  private recorder: MediaRecorder | null = null
  private chunks: Blob[] = []
  private stopped_promise: Promise<void> | null = null
  private stopped_resolve: (() => void) | null = null
  private file_name: string = ''

  constructor (renderer: THREE.WebGLRenderer, preview_width: number, preview_height: number) {
    this.renderer_ref = renderer
    this.width = preview_width
    this.height = preview_height

    console.log(this.renderer_ref.domElement.height, this.renderer_ref.domElement.width)
  }

  public start (saved_name: string): void {
    this.old_size = this.renderer_ref.getSize(new THREE.Vector2())
    this.old_pr = this.renderer_ref.getPixelRatio()
    this.renderer_ref.setPixelRatio(1)
    this.renderer_ref.setSize(this.width, this.height, false)
    this.file_name = saved_name

    const stream: MediaStream = this.renderer_ref.domElement.captureStream(this.fps)
    this.recorder = new MediaRecorder(stream, {
      mimeType: this.supported_webm_mime_type(),
      videoBitsPerSecond: this.kbps * 1000
    })
    this.chunks = []
    this.recorder.ondataavailable = e => (e.data.size !== 0) && this.chunks.push(e.data)
    this.stopped_promise = new Promise<void>(resolve => {
      this.stopped_resolve = resolve
      this.recorder!.onstop = () => { resolve() }
    })
    this.recorder.start()
  }

  public async stop (): Promise<File> {
    if (!this.recorder) throw new Error('Recorder not started')
    this.recorder.stop()
    if (this.stopped_promise) {
      await this.stopped_promise
    }
    // restore the renderer size and pixel ratio
    if (this.old_size && this.old_pr !== null) {
      this.renderer_ref.setSize(this.old_size.x, this.old_size.y, false)
      this.renderer_ref.setPixelRatio(this.old_pr)
    }
    const file = new File(this.chunks, this.file_name, { type: this.supported_webm_mime_type() })
    this.recorder = null
    this.stopped_promise = null
    this.stopped_resolve = null
    this.chunks = []
    return file
  }

  private supported_webm_mime_type (): string {
    const prefs = ['video/webm;codecs=av01', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    return prefs.find(m => MediaRecorder.isTypeSupported(m)) ?? 'video/webm'
  }
}
