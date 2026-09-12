package com.agentruntime

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Reads this process's own logcat buffer. An app may read its own log lines
 * without any permission, which makes the native engine's real error text
 * (llama.rn's LOG_ERROR lines, libc++ messages) reachable from the in-app
 * diagnostics log — the JS layer otherwise sees only "Unknown error".
 */
class LogcatModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
  override fun getName(): String = "Logcat"

  @ReactMethod
  fun dump(maxLines: Int, promise: Promise) {
    try {
      val pid = android.os.Process.myPid().toString()
      val proc = Runtime.getRuntime().exec(arrayOf("logcat", "-d", "-v", "time", "--pid=$pid", "-t", maxLines.toString()))
      val reader = BufferedReader(InputStreamReader(proc.inputStream))
      val sb = StringBuilder()
      var line = reader.readLine()
      while (line != null) {
        sb.append(line).append('\n')
        line = reader.readLine()
      }
      reader.close()
      proc.waitFor()
      promise.resolve(sb.toString())
    } catch (e: Exception) {
      promise.reject("logcat_failed", e.message, e)
    }
  }
}
