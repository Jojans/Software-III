package co.edu.demoacademico.controller;

import co.edu.demoacademico.exception.EmailYaRegistradoException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.Map;

@ControllerAdvice
public class GlobalExceptionHandler {
    @ExceptionHandler(EmailYaRegistradoException.class)
    public ResponseEntity<?> manejarEmailDuplicado(EmailYaRegistradoException ex){
        return ResponseEntity.status(HttpStatus.CONFLICT).body(
                Map.of(
                        "timestamp", LocalDateTime.now(),
                        "status", 409,
                        "error", "Conflicto",
                        "message", ex.getMessage()
                )
        );
    }
}
