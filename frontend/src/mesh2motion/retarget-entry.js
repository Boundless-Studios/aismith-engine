import '../../../vendor/mesh2motion-app/src/styles.css'
import '../../../vendor/mesh2motion-app/src/animation-player.css'
import '../../../vendor/mesh2motion-app/src/settings-dropdown.css'
import '../../../vendor/mesh2motion-app/src/dialog.css'
import '../../../vendor/mesh2motion-app/src/offline.css'
import '../../../vendor/mesh2motion-app/src/retarget/retarget.css'
import { RetargetModule } from '../../../vendor/mesh2motion-app/src/retarget/retarget.ts'
import { resolveRetargetLaunch } from './RetargetLaunchContract.ts'

document.addEventListener('DOMContentLoaded', () => {
  new RetargetModule().init(resolveRetargetLaunch(window.location.search, window.location.hostname))
})
