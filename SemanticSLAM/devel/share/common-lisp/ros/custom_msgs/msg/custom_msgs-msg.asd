
(cl:in-package :asdf)

(defsystem "custom_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ObjectList" :depends-on ("_package_ObjectList"))
    (:file "_package_ObjectList" :depends-on ("_package"))
    (:file "WorldObject" :depends-on ("_package_WorldObject"))
    (:file "_package_WorldObject" :depends-on ("_package"))
  ))