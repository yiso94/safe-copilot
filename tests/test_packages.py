def test_flash_attn():
    import flash_attn

    assert flash_attn.__version__ == "2.8.3"
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

    assert pad_input is not None
    assert unpad_input is not None
    assert flash_attn_varlen_qkvpacked_func is not None
