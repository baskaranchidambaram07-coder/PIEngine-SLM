package com.agentruntime

import android.content.Intent
import android.provider.CalendarContract
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

/**
 * Hands a schedule_event tool call off to the phone's own Calendar app via
 * Intent.ACTION_INSERT — the Calendar app does the actual write, so this
 * declares no permission and inserts nothing silently. The user reviews the
 * pre-filled event and taps Save, the same "draft, never auto-sent" shape
 * as draft_email's mailto: handoff. React Native's Linking module cannot
 * express this: it only opens URL-scheme intents (mailto:, tel:, geo:, ...),
 * not an explicit Intent with typed long extras, hence this small module.
 */
class CalendarModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
  override fun getName(): String = "CalendarInsert"

  @ReactMethod
  fun openInsert(title: String, description: String, startMillis: Double, endMillis: Double, promise: Promise) {
    try {
      val intent = Intent(Intent.ACTION_INSERT)
        .setData(CalendarContract.Events.CONTENT_URI)
        .putExtra(CalendarContract.Events.TITLE, title)
        .putExtra(CalendarContract.Events.DESCRIPTION, description)
        .putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, startMillis.toLong())
        .putExtra(CalendarContract.EXTRA_EVENT_END_TIME, endMillis.toLong())
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
      reactApplicationContext.startActivity(intent)
      promise.resolve(true)
    } catch (e: Exception) {
      promise.reject("calendar_insert_failed", e.message, e)
    }
  }
}
