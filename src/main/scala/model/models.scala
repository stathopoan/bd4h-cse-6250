package model

import java.sql.Date

case class Note(patientID: Integer, hadmID: Integer, category: String, description: String, text: String)
case class Diag(RowId:Integer, patientID: Integer, hadmID: Integer, seqNum: String, icd9Code: String)
case class Procedure(RowId:Integer, patientID: Integer, hadmID: Integer, seqNum: String, icd9Code: String)
