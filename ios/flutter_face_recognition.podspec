Pod::Spec.new do |s|
  s.name             = 'flutter_face_recognition'
  s.version          = '1.0.0'
  s.summary          = 'Offline Face Recognition, Cosine Similarity & Multi-Scale Anti-Spoofing plugin for Flutter.'
  s.description      = <<-DESC
High-performance offline Face Recognition (MobileFaceNet 192-dim), Cosine Similarity matching,
and Multi-Scale Passive Liveness Detection (MiniFASNet) for Flutter using TFLite and Apple Vision.
                       DESC
  s.homepage         = 'https://github.com/Andrew-Hardianto/flutter_face_recognition'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Andrew Hardianto' => 'andrew@example.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.dependency 'TensorFlowLiteSwift', '~> 2.14.0'
  s.platform         = :ios, '15.0'

  # TensorFlowLiteSwift requires static framework linkage on CocoaPods
  s.static_framework = true

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }
  s.swift_version = '5.9'
  s.resource_bundles = {
    'flutter_face_recognition_privacy' => ['Classes/PrivacyInfo.xcprivacy']
  }
end

