(function ($) {
  "use strict";

  // =========================
  // 🔹 NICE SELECT
  // =========================
  $(document).ready(function () {
    $("select").niceSelect();
  });

  // =========================
  // 🔹 FIXED NAVBAR SCROLL
  // =========================
  $(window).scroll(function () {
    var window_top = $(window).scrollTop() + 1;

    if (window_top > 50) {
      $(".main_menu").addClass("menu_fixed animated fadeInDown");
    } else {
      $(".main_menu").removeClass("menu_fixed animated fadeInDown");
    }
  });

  // =========================
  // 🔹 OWL CAROUSEL (REVIEWS)
  // =========================
  var review = $(".client_review_part");

  if (review.length) {
    review.owlCarousel({
      items: 1,
      loop: true,
      dots: true,
      autoplay: true,
      autoplayHoverPause: true,
      autoplayTimeout: 5000,
      nav: false,
      smartSpeed: 2000,
    });
  }

  // =========================
  // 🔹 MAILCHIMP
  // =========================
  function mailChimp() {
    $("#mc_embed_signup").find("form").ajaxChimp();
  }
  mailChimp();

  // =========================================================
  // 🔥 NEW PREMIUM FEATURES (ADDED WITHOUT BREAKING OLD CODE)
  // =========================================================

  // =========================
  // 🔹 SMOOTH SCROLL
  // =========================
  $('a[href^="#"]').on("click", function (e) {
    e.preventDefault();

    var target = $(this.getAttribute("href"));

    if (target.length) {
      $("html, body")
        .stop()
        .animate(
          {
            scrollTop: target.offset().top - 50,
          },
          800,
        );
    }
  });

  // =========================
  // 🔹 CARD HOVER EFFECT
  // =========================
  $(".stats-card, .disease-card").hover(
    function () {
      $(this).css({
        transform: "translateY(-8px)",
        transition: "0.3s",
      });
    },
    function () {
      $(this).css({
        transform: "translateY(0)",
      });
    },
  );

  // =========================
  // 🔹 COUNTER ANIMATION (STATS)
  // =========================
  $(".stats-card h2").each(function () {
    var $this = $(this);
    var countTo = parseInt($this.text());

    if (!isNaN(countTo)) {
      $({ countNum: 0 }).animate(
        { countNum: countTo },
        {
          duration: 2000,
          easing: "swing",
          step: function () {
            $this.text(Math.floor(this.countNum) + "+");
          },
          complete: function () {
            $this.text(this.countNum + "+");
          },
        },
      );
    }
  });

  // =========================
  // 🔹 HERO FADE ANIMATION
  // =========================
  $(".hero-section h1").hide().fadeIn(1200);
  $(".hero-section p").hide().delay(300).fadeIn(1200);
  $(".hero-btns").hide().delay(600).fadeIn(1200);
})(jQuery);
