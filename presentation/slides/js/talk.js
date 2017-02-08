$(function() {
	renderMathInElement(document.body);

	var obj = impress();
	obj.init();
	initSlideNo(obj);

	$(".inline-math").each(function() {
		var expr = $(this).text();
		$(this).text("");
		katex.render( expr, $(this).get(0) );
	});

});

