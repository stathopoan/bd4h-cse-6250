package model

import java.sql.Date

case class Note(patientID: String, hadmID: String, category: String, description: String, text: String)
case class Diag(patientID: String, hadmID: String, seqNum: String, icd9Code: String)
case class Procedure(patientID: String, hadmID: String, seqNum: String, icd9Code: String)
