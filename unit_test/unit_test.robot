*** Settings ***
Library    ../src/utils.py

*** Variables ***
${img_path}    ../images/test_images/
${img_name}    sample1.jpg

*** Tasks ***
Unit Test
    [Documentation]    Unit Test For OCR Model
    ${text}    ${enhanced_image_path}=    utils.Read Text From Image    image_path=${img_path}${img_name}
    # ${text}    ${files}=    Get Curwd
    Log To Console    \nText: ${text}
    Log To Console    enhanced_image_path: ${enhanced_image_path}
    utils.Log Image    img_path=${enhanced_image_path}
