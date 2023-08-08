
(cl:in-package :asdf)

(defsystem "custom_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "ClearLaser" :depends-on ("_package_ClearLaser"))
    (:file "_package_ClearLaser" :depends-on ("_package"))
  ))