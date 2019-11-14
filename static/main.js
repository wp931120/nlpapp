var Main = {
    init: function () {
        var _this = this;
        $("#doc").html("吴恩达（1976-，英文名：Andrew Ng），华裔美国人，是斯坦福大学计算机科学系和电子工程系副教授，人工智能实验室主任。吴恩达是人工智能和机器学习领域国际上最权威的学者之一。吴恩达也是在线教育平台Coursera的联合创始人（with Daphne Koller）。\n" +
            "2014年5月16日，吴恩达加入百度，担任百度公司首席科学家，负责百度研究院的领导工作，尤其是Baidu Brain计划。 [1] \n" +
            "2017年10月，吴恩达将出任Woebot公司新任董事长，该公司拥有一款同名聊天机器人。")
        $("#qry").html("吴恩达的英文名是什么？")
        $("#cihead").html("菩萨蛮:")

        $("#gen_ans").click(function () {
            _this.machine_reading(this)
        });
        $("#gen_ci").click(function () {
            _this.generate_ci(this)
        });

    },
    machine_reading: function () {
        var doc = $("#doc").val();
        var qry = $("#qry").val();
        $.ajax({
            url: "/gen_ans",
            type: 'POST',
            data : {'doc':doc,
                    'qry':qry,
            },
            dataType:"json",
            beforeSend: function () {
                if (doc == "" || qry=="" ){
                    alert('请输入资料和问题')
                    return false
                }else {
                    $('#ans').html("答案生成中")
                }
            },
            error :function(){
                alert('error')
            },
            success: function (data) {
                 console.log(data)
                 $('#ans').html(data.content)
            },
        });
    },
    generate_ci: function () {
        var ci_head = $("#cihead").val();
        var topk = $("#topk").val();
        $.ajax({
            url: "/gen_ci",
            type: 'POST',
            data : {'ci_head':ci_head,
                    'topk':topk,
            },
            dataType:"json",
            beforeSend: function () {
                if (ci_head == ""||topk==""){
                    alert('请输入词牌名和topk')
                    return false
                }else {
                    $("#cibody").html("词生成中")
                }
            },
            error :function(){
                alert('error')
            },
            success: function (data) {
                 console.log(data)
                 $("#cibody").html(data.content)
            },
        });
    },
}

$(function(){
   Main.init();
});
