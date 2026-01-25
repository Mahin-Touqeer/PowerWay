import styled from "@emotion/styled";
import { Switch } from "@mui/material";
import { useState } from "react";

const IOSSwitch = styled((props) => (
  <Switch focusVisibleClassName=".Mui-focusVisible" disableRipple {...props} />
))(() => ({
  width: 42,
  height: 24,
  padding: 0,
  "& .MuiSwitch-switchBase": {
    padding: 2,
    "&.Mui-checked": {
      transform: "translateX(18px)",
      color: "#fff",
      "& + .MuiSwitch-track": {
        backgroundColor: "#39E079", // green like your image
        opacity: 1,
        border: 0,
      },
    },
  },
  "& .MuiSwitch-thumb": {
    width: 20,
    height: 20,
  },
  "& .MuiSwitch-track": {
    borderRadius: 12,
    backgroundColor: "#ccc",
    opacity: 1,
  },
}));

export default function ToggleButton({ value, changeFunction }) {
  const handleChange = (event) => {
    changeFunction((prev) => ({
      ...prev,
      availaible: event.target.checked,
    }));
  };

  return <IOSSwitch checked={value?.availaible} onChange={handleChange} />;
}
